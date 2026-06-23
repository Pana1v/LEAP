"""
Experiment: can a smaller-data-trained GNN guide LEAP at N=200?

We test three model checkpoints, all same architecture, all evaluated on the
same N=200 val subset (seed=42, first 20 scenarios):
  - gnn_10obj_best.pt   (trained on dataset_10)
  - gnn_final_40obj.pt  (trained on dataset_40)
  - gnn_final_200obj.pt (baseline; trained on dataset_200)

Question: does GNN inference get faster with the smaller-data model, and does
the smaller model still produce logits good enough for LEAP to stay within the
0.06% optimality budget at N=200?

Hypothesis: inference time is architecture-bound (same for all three);
quality drops as training-N decreases.

Run from src/:
  /home/pan-navigator/binning_venv/bin/python experiment_small_model_at_n200.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from gnn_ilp_circuit import (
    solve_circuit_cold,
    solve_circuit_pruned,
    load_scenarios,
    prepare_scenario,
    gnn_rollout_with_logits,
)
from gnn_gui import load_model
from gnn_train import split_dataset

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "small_model_at_n200.json"

MODELS = {
    "n10":            REPO / "models" / "gnn_10obj_best.pt",
    "n40":            REPO / "models" / "gnn_final_40obj.pt",
    "n200":           REPO / "models" / "gnn_final_200obj.pt",
    "thin_h64_n40":   REPO / "models" / "gnn_thin_h64_n40_curr_to_40.pt",
}

VAL_SPLIT = 0.1
VAL_SEED = 42
N_SCENARIOS = 20
LEAP_K = 15
N_WARMUP = 10


def time_leap(scenarios, model, device):
    gnn_times, total_times, leap_costs, opt_costs = [], [], [], []
    # Warmup
    for s in scenarios[:N_WARMUP]:
        st = prepare_scenario(s, device)
        _, gnn_seq, logits = gnn_rollout_with_logits(model, st, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        solve_circuit_pruned(s, gnn_seq, logits, k_neighbors=LEAP_K,
                             max_objects=len(s["objects"]) + 2)
    # Timed
    for s in scenarios:
        t0 = time.time()
        st = prepare_scenario(s, device)
        t_gnn0 = time.time()
        _, gnn_seq, logits = gnn_rollout_with_logits(model, st, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gnn = time.time() - t_gnn0
        leap_cost, _, _ = solve_circuit_pruned(
            s, gnn_seq, logits, k_neighbors=LEAP_K,
            max_objects=len(s["objects"]) + 2,
        )
        total = time.time() - t0
        gnn_times.append(t_gnn * 1000)
        total_times.append(total * 1000)
        leap_costs.append(leap_cost)
        opt_cost, _ = solve_circuit_cold(s, max_objects=len(s["objects"]) + 2)
        opt_costs.append(opt_cost)
    gnn_times = np.array(gnn_times)
    total_times = np.array(total_times)
    leap_costs = np.array(leap_costs)
    opt_costs = np.array(opt_costs)
    gaps = (leap_costs - opt_costs) / opt_costs * 100
    return {
        "gnn_inference_ms_mean": float(gnn_times.mean()),
        "gnn_inference_ms_median": float(np.median(gnn_times)),
        "total_leap_ms_mean": float(total_times.mean()),
        "total_leap_ms_median": float(np.median(total_times)),
        "leap_cost_mean": float(leap_costs.mean()),
        "opt_cost_mean": float(opt_costs.mean()),
        "gap_mean_pct": float(gaps.mean()),
        "gap_max_pct": float(gaps.max()),
        "gap_min_pct": float(gaps.min()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    scenarios = load_scenarios(REPO / "data" / "dataset_200_objects.json")
    _, val = split_dataset(scenarios, VAL_SPLIT, VAL_SEED)
    val = val[:N_SCENARIOS]
    print(f"N=200, {len(val)} val scenarios\n")

    results = {"config": {"n_scenarios": len(val), "k": LEAP_K, "device": str(device)},
               "models": {}}

    for label, model_path in MODELS.items():
        if not model_path.exists():
            print(f"[skip] {label}: {model_path} missing")
            continue
        print(f"=== model={label} ({model_path.name}) ===")
        model = load_model(device, str(model_path))
        r = time_leap(val, model, device)
        results["models"][label] = {"model_path": str(model_path), **r}
        print(f"  GNN inference: mean {r['gnn_inference_ms_mean']:.1f} ms, "
              f"median {r['gnn_inference_ms_median']:.1f} ms")
        print(f"  Total LEAP:    mean {r['total_leap_ms_mean']:.1f} ms, "
              f"median {r['total_leap_ms_median']:.1f} ms")
        print(f"  Optimality gap: mean {r['gap_mean_pct']:.4f}%, "
              f"max {r['gap_max_pct']:.4f}%")
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
