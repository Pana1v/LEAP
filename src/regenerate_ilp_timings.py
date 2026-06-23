"""
Regenerate CP-SAT Circuit + LEAP timings on paper-matching validation subsets.

Replaces the MTZ-era timing numbers in v8 Tables IV and VI. Costs are exact-
solver-invariant (verified separately by scripts/verify_ilp_circuit_equivalence.py);
this script only re-measures wall-clock inference time.

Run from src/:
    /home/pan-navigator/binning_venv/bin/python regenerate_ilp_timings.py
"""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from gnn_ilp_circuit import (
    solve_circuit_cold,
    solve_circuit_pruned,
    select_model_by_count,
    load_scenarios,
    prepare_scenario,
    gnn_rollout_with_logits,
)
from gnn_gui import load_model
from gnn_train import split_dataset

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "experiments" / "ilp_timing_circuit_v8.json"

# Paper-matching sample sizes (see plan + v8 captions).
SAMPLE_SIZES = {
    5: 50,
    10: 50,
    20: 50,
    40: 50,
    100: 50,
    200: 20,
}

# Validation split — same as gnn_train.split_dataset (seed=42, 10% val).
VAL_SPLIT = 0.1
VAL_SEED = 42

# Warmup passes before timing each N (separate per solver to avoid cross-warmup
# from a different problem size affecting JIT/cache state).
N_WARMUP = 10

# LEAP top-k arcs per node.
LEAP_K = 15


def get_val(scenarios: List[Dict]) -> List[Dict]:
    _, val = split_dataset(scenarios, VAL_SPLIT, VAL_SEED)
    return val


def stats(times: List[float]) -> Dict[str, float]:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "n": int(arr.size),
    }


def time_unpruned(scenarios: List[Dict]) -> Dict:
    times_ms: List[float] = []
    costs: List[float] = []
    # Warmup
    warm = scenarios[: min(N_WARMUP, len(scenarios))]
    for s in warm:
        solve_circuit_cold(s, max_objects=len(s["objects"]) + 2)
    # Timed
    for s in scenarios:
        cost, elapsed_s = solve_circuit_cold(s, max_objects=len(s["objects"]) + 2)
        times_ms.append(elapsed_s * 1000.0)
        costs.append(cost)
    return {"timing": stats(times_ms), "costs": costs}


def time_leap(scenarios: List[Dict], model, device) -> Dict:
    times_ms: List[float] = []
    costs: List[float] = []
    # Warmup
    warm = scenarios[: min(N_WARMUP, len(scenarios))]
    for s in warm:
        st = prepare_scenario(s, device)
        _, gnn_seq, logits = gnn_rollout_with_logits(model, st, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        solve_circuit_pruned(s, gnn_seq, logits, k_neighbors=LEAP_K, max_objects=len(s["objects"]) + 2)
    # Timed (GNN rollout + pruned solve = LEAP total wall time)
    for s in scenarios:
        t0 = time.time()
        st = prepare_scenario(s, device)
        _, gnn_seq, logits = gnn_rollout_with_logits(model, st, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        cost, elapsed_solver, _n_arcs = solve_circuit_pruned(
            s, gnn_seq, logits, k_neighbors=LEAP_K, max_objects=len(s["objects"]) + 2
        )
        total_ms = (time.time() - t0) * 1000.0
        times_ms.append(total_ms)
        costs.append(cost)
    return {"timing": stats(times_ms), "costs": costs}


def gap_pct(leap_costs: List[float], optimal_costs: List[float]) -> Dict[str, float]:
    deltas = [(l - o) / o * 100.0 for l, o in zip(leap_costs, optimal_costs)]
    return {
        "mean_pct": float(np.mean(deltas)),
        "max_pct": float(np.max(deltas)),
        "min_pct": float(np.min(deltas)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, nargs="+", default=sorted(SAMPLE_SIZES.keys()))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "val_split": VAL_SPLIT,
            "val_seed": VAL_SEED,
            "n_warmup": N_WARMUP,
            "leap_k": LEAP_K,
            "device": str(device),
        },
        "per_n": {},
    }

    for n in args.ns:
        k = SAMPLE_SIZES.get(n)
        if k is None:
            print(f"[skip] N={n}: no configured sample size")
            continue
        dataset_path = DATA / f"dataset_{n}_objects.json"
        if not dataset_path.exists():
            print(f"[skip] N={n}: {dataset_path} missing")
            continue
        print(f"\n=== N={n} (sample={k}) ===")
        scenarios = load_scenarios(dataset_path)
        val = get_val(scenarios)
        subset = val[:k]
        print(f"loaded {len(scenarios)} total, val={len(val)}, taking first {len(subset)}")

        # Unpruned
        t0 = time.time()
        unpruned = time_unpruned(subset)
        print(f"  unpruned: mean={unpruned['timing']['mean_ms']:.1f}ms "
              f"median={unpruned['timing']['median_ms']:.1f}ms "
              f"p95={unpruned['timing']['p95_ms']:.1f}ms (took {time.time()-t0:.1f}s)")

        # LEAP
        model_path = select_model_by_count(n)
        if not model_path.exists():
            print(f"  [warn] no GNN model at {model_path}; skipping LEAP for N={n}")
            leap = None
            speedup = None
            max_gap = None
        else:
            model = load_model(device, str(model_path))
            t0 = time.time()
            leap = time_leap(subset, model, device)
            print(f"  LEAP:     mean={leap['timing']['mean_ms']:.1f}ms "
                  f"median={leap['timing']['median_ms']:.1f}ms "
                  f"p95={leap['timing']['p95_ms']:.1f}ms (took {time.time()-t0:.1f}s)")
            speedup = unpruned["timing"]["mean_ms"] / leap["timing"]["mean_ms"]
            gap = gap_pct(leap["costs"], unpruned["costs"])
            max_gap = gap["max_pct"]
            print(f"  speedup={speedup:.2f}x  max gap={max_gap:.4f}%  mean gap={gap['mean_pct']:.4f}%")

        results["per_n"][str(n)] = {
            "n_scenarios": k,
            "unpruned": unpruned,
            "leap": leap,
            "speedup": speedup,
            "max_gap_pct": max_gap,
        }

        # Write after each N so partial progress is saved
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nFinal: {OUT}")


if __name__ == "__main__":
    main()
