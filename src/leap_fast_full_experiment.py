"""Full LEAP (fast GNN rollout + CP-SAT pruned) at N∈{10,40,200} for the three
best candidate models. Output drops into experiments/leap_fast_v8.json."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from gnn_ilp_circuit import (
    load_scenarios,
    prepare_scenario,
    solve_circuit_cold,
    solve_circuit_pruned,
)
from gnn_gui import load_model
from gnn_train import split_dataset
from fast_rollout import fast_rollout_with_logits

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "leap_fast_v8.json"

VAL_SPLIT = 0.1
VAL_SEED = 42
LEAP_K = 15
N_WARMUP = 10
SAMPLE_BY_N = {10: 50, 40: 50, 200: 20}

MODELS = [
    ("n40_h128",     REPO / "models" / "gnn_final_40obj.pt"),
    ("thin_h64_n40", REPO / "models" / "gnn_thin_h64_n40_curr_to_40.pt"),
    ("n200_h128",    REPO / "models" / "gnn_final_200obj.pt"),
]


def time_model_at_n(n, k, model, device):
    sc_raw = load_scenarios(REPO / "data" / f"dataset_{n}_objects.json")
    _, val = split_dataset(sc_raw, VAL_SPLIT, VAL_SEED)
    val_raw = val[:k]
    val_t = [prepare_scenario(s, device) for s in val_raw]

    # Warmup
    for sr, st in list(zip(val_raw, val_t))[:N_WARMUP]:
        _, gseq, logits = fast_rollout_with_logits(model, st, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        solve_circuit_pruned(sr, gseq, logits, k_neighbors=LEAP_K,
                             max_objects=len(sr["objects"]) + 2)

    gnn_ms, total_ms, leap_costs, opt_costs = [], [], [], []
    for sr, st in zip(val_raw, val_t):
        t0 = time.time()
        tg0 = time.time()
        _, gseq, logits = fast_rollout_with_logits(model, st, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tg = time.time() - tg0
        cost, _, _ = solve_circuit_pruned(sr, gseq, logits, k_neighbors=LEAP_K,
                                          max_objects=len(sr["objects"]) + 2)
        total = time.time() - t0
        gnn_ms.append(tg * 1000)
        total_ms.append(total * 1000)
        leap_costs.append(cost)
        opt, _ = solve_circuit_cold(sr, max_objects=len(sr["objects"]) + 2)
        opt_costs.append(opt)

    gaps = (np.array(leap_costs) - np.array(opt_costs)) / np.array(opt_costs) * 100
    return {
        "n": n,
        "k_scenarios": k,
        "gnn_ms_mean": float(np.mean(gnn_ms)),
        "gnn_ms_median": float(np.median(gnn_ms)),
        "total_ms_mean": float(np.mean(total_ms)),
        "total_ms_median": float(np.median(total_ms)),
        "gap_mean_pct": float(np.mean(gaps)),
        "gap_max_pct": float(np.max(gaps)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    results = {}
    for label, path in MODELS:
        if not path.exists():
            print(f"[skip] {label}"); continue
        model = load_model(device, str(path))
        results[label] = {}
        for n, k in SAMPLE_BY_N.items():
            print(f"=== {label}  N={n}  (k={k}) ===")
            r = time_model_at_n(n, k, model, device)
            results[label][str(n)] = r
            print(f"  GNN: median {r['gnn_ms_median']:.1f} ms")
            print(f"  Total LEAP: median {r['total_ms_median']:.1f} ms")
            print(f"  Gap: mean {r['gap_mean_pct']:.4f}%, max {r['gap_max_pct']:.4f}%\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
