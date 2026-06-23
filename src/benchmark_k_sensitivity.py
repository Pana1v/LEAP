"""
Ablation study: k-sensitivity for GNN-guided arc pruning.

Sweeps k (neighbourhood size) across multiple problem sizes and measures
solve time, optimality gap, and number of arcs.

Run from src/:
  python3 benchmark_k_sensitivity.py --max-scenarios 20
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from gnn_train import (
    FEATURE_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_DROPOUT,
    GNNPolicy,
    load_scenarios,
    prepare_scenario,
    split_dataset,
)
from gnn_ilp_circuit import (
    build_cost_matrix,
    gnn_rollout_with_logits,
    solve_circuit_cold,
    solve_circuit_pruned,
    solve_cbc_cold,
)

SEED = 42
VAL_SPLIT = 0.1
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

K_VALUES = [3, 5, 10, 15, 20, 30, 50]


def load_gnn_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hdim = ckpt.get("hidden_dim", DEFAULT_HIDDEN_DIM)
    heads = ckpt.get("heads", DEFAULT_ATTENTION_HEADS)
    drop = ckpt.get("dropout", DEFAULT_DROPOUT)
    model = GNNPolicy(FEATURE_DIM, hdim, heads, drop)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def run_k_sweep(dataset_path, model_path, max_scenarios, device):
    """Run k-sensitivity sweep on one dataset."""
    all_scenarios = load_scenarios(Path(dataset_path))
    _, val_raw = split_dataset(all_scenarios, VAL_SPLIT, SEED)
    val_raw = val_raw[:max_scenarios]
    N = len(val_raw[0]["objects"])

    model = load_gnn_model(model_path, device)

    # Pre-compute GNN rollouts with logits for all scenarios
    gnn_data = []
    for s_raw in val_raw:
        s = prepare_scenario(s_raw, device)
        with torch.no_grad():
            _, seq, logits = gnn_rollout_with_logits(model, s, device)
        gnn_data.append((s_raw, seq, logits))

    results = {"N": N, "n_scenarios": len(val_raw), "k_results": {}}

    max_obj = N + 10  # allow headroom beyond object count

    # Baseline: cold circuit (unpruned)
    print(f"  Running cold circuit (unpruned)...")
    cold_costs, cold_times = [], []
    for s_raw, _, _ in gnn_data:
        cost, t = solve_circuit_cold(s_raw, max_objects=max_obj)
        cold_costs.append(cost)
        cold_times.append(t)
    results["cold_circuit"] = {
        "mean_cost": float(np.mean(cold_costs)),
        "mean_time": float(np.mean(cold_times)),
    }
    print(f"    Cold circuit: cost={np.mean(cold_costs):.1f}, time={np.mean(cold_times):.3f}s")

    # Baseline: CBC cold (skip for N>60 — too slow)
    if N <= 60:
        print(f"  Running CBC cold...")
        cbc_costs, cbc_times = [], []
        for s_raw, _, _ in gnn_data:
            cost, t = solve_cbc_cold(s_raw, max_objects=max_obj)
            cbc_costs.append(cost)
            cbc_times.append(t)
        results["cbc_cold"] = {
            "mean_cost": float(np.mean(cbc_costs)),
            "mean_time": float(np.mean(cbc_times)),
        }
        print(f"    CBC cold: cost={np.mean(cbc_costs):.1f}, time={np.mean(cbc_times):.3f}s")
    else:
        cbc_times = cold_times  # fallback for speedup calc
        results["cbc_cold"] = {"mean_cost": 0.0, "mean_time": 0.0, "skipped": True}
        print(f"    CBC cold: SKIPPED (N={N} > 60, too slow)")

    # Sweep k values
    for k in K_VALUES:
        if k >= N:
            # k >= N means no pruning, skip (equivalent to cold)
            continue
        print(f"  k={k}...")
        pruned_costs, pruned_times, arc_counts = [], [], []
        for s_raw, seq, logits in gnn_data:
            cost, t, n_arcs = solve_circuit_pruned(s_raw, seq, logits, k_neighbors=k, max_objects=max_obj)
            pruned_costs.append(cost)
            pruned_times.append(t)
            arc_counts.append(n_arcs)

        # Compute gaps vs cold circuit
        gaps = []
        for pc, cc in zip(pruned_costs, cold_costs):
            if cc > 0:
                gaps.append((pc - cc) / cc * 100.0)

        speedup_vs_cold = np.mean(cold_times) / np.mean(pruned_times) if np.mean(pruned_times) > 0 else 0
        speedup_vs_cbc = np.mean(cbc_times) / np.mean(pruned_times) if np.mean(pruned_times) > 0 else 0

        results["k_results"][str(k)] = {
            "mean_cost": float(np.mean(pruned_costs)),
            "mean_time": float(np.mean(pruned_times)),
            "mean_arcs": float(np.mean(arc_counts)),
            "mean_gap_vs_cold_pct": float(np.mean(gaps)) if gaps else 0.0,
            "max_gap_vs_cold_pct": float(max(gaps)) if gaps else 0.0,
            "speedup_vs_cold": float(speedup_vs_cold),
            "speedup_vs_cbc": float(speedup_vs_cbc),
            "n_exact": sum(1 for g in gaps if abs(g) < 0.001),
        }
        print(f"    k={k}: cost={np.mean(pruned_costs):.1f}, time={np.mean(pruned_times):.3f}s, "
              f"arcs={np.mean(arc_counts):.0f}, gap={np.mean(gaps):.4f}%, "
              f"max_gap={max(gaps):.4f}%, speedup_vs_cbc={speedup_vs_cbc:.1f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description="k-sensitivity ablation for arc pruning")
    parser.add_argument("--max-scenarios", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    datasets = [
        ("N=40",  DATA_DIR / "dataset_40_objects.json",  MODELS_DIR / "gnn_final_40obj.pt"),
        ("N=100", DATA_DIR / "dataset_100_objects.json", MODELS_DIR / "gnn_final_40obj.pt"),
        ("N=200", DATA_DIR / "dataset_200_objects.json", MODELS_DIR / "gnn_final_200obj.pt"),
    ]

    all_results = {}
    for label, ds_path, model_path in datasets:
        if not ds_path.exists() or not model_path.exists():
            print(f"Skipping {label}: missing files")
            continue
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
        all_results[label] = run_k_sweep(str(ds_path), str(model_path), args.max_scenarios, device)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Dataset':<8} {'k':>4} {'Cost':>10} {'Time(s)':>10} {'Arcs':>8} {'Gap%':>8} {'MaxGap%':>8} {'vs CBC':>8}")
    print(f"{'-'*90}")
    for label, res in all_results.items():
        # Cold circuit row
        cc = res["cold_circuit"]
        cbc = res["cbc_cold"]
        print(f"{label:<8} {'full':>4} {cc['mean_cost']:>10.1f} {cc['mean_time']:>10.3f} {'---':>8} {'0.000':>8} {'0.000':>8} {cbc['mean_time']/cc['mean_time']:>7.1f}x")
        for k_str in sorted(res["k_results"].keys(), key=int):
            r = res["k_results"][k_str]
            print(f"{label:<8} {k_str:>4} {r['mean_cost']:>10.1f} {r['mean_time']:>10.3f} {r['mean_arcs']:>8.0f} {r['mean_gap_vs_cold_pct']:>7.4f}% {r['max_gap_vs_cold_pct']:>7.4f}% {r['speedup_vs_cbc']:>7.1f}x")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
