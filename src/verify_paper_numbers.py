"""
Spot-check all numerical claims in the paper against actual model outputs.

Loads each GNN checkpoint, runs rollout on validation scenarios, and compares
against paper Table claims (gap vs greedy, win-rate, gap vs ILP).

Run from src/:  python3 verify_paper_numbers.py
"""

import json
import sys
from pathlib import Path

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
    rollout_model,
    split_dataset,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
VAL_SPLIT = 0.1


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hdim = ckpt.get("hidden_dim", DEFAULT_HIDDEN_DIM)
    heads = ckpt.get("heads", DEFAULT_ATTENTION_HEADS)
    drop = ckpt.get("dropout", DEFAULT_DROPOUT)
    model = GNNPolicy(FEATURE_DIM, hdim, heads, drop)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_checkpoint(model_path, dataset_path, device, ilp_key=None):
    """Evaluate a model on the validation split, returning paper-style metrics."""
    model = load_model(model_path, device)
    scenarios = load_scenarios(dataset_path)
    _, val_raw = split_dataset(scenarios, VAL_SPLIT, SEED)
    val = [prepare_scenario(s, device) for s in val_raw]

    costs, greedy_costs, gaps = [], [], []
    ilp_gaps = []
    win_count = 0

    with torch.no_grad():
        for s in val:
            cost = rollout_model(model, s, device)
            gc = s["greedy_cost"]
            costs.append(cost)
            greedy_costs.append(gc)
            gap = (gc - cost) / gc * 100.0 if gc > 0 else 0.0
            gaps.append(gap)
            if cost < gc:
                win_count += 1

            if ilp_key:
                ilp_cost = s["ilp_costs"].get(ilp_key)
                if ilp_cost is not None:
                    ilp_gaps.append((cost - ilp_cost) / ilp_cost * 100.0)

    n = len(val)
    return {
        "n_scenarios": n,
        "mean_gnn_cost": float(np.mean(costs)),
        "mean_greedy_cost": float(np.mean(greedy_costs)),
        "gap_vs_greedy_pct": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps)),
        "win_rate_pct": win_count / n * 100.0,
        "mean_ilp_gap_pct": float(np.mean(ilp_gaps)) if ilp_gaps else None,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    checks = [
        {
            "label": "N=10",
            "model": MODELS_DIR / "gnn_10obj_best.pt",
            "dataset": DATA_DIR / "dataset_10_objects.json",
            "ilp_key": "10",
            "paper_claims": {
                "gap_vs_greedy": 5.81,
                "win_rate": 90.8,
                "ilp_gap": 1.81,
            },
        },
        {
            "label": "N=40",
            "model": MODELS_DIR / "gnn_40obj_best.pt",
            "dataset": DATA_DIR / "dataset_40_objects.json",
            "ilp_key": None,
            "paper_claims": {
                "gap_vs_greedy": 3.23,
                "win_rate": 98.0,
            },
        },
        {
            "label": "N=200",
            "model": MODELS_DIR / "gnn_200obj_best.pt",
            "dataset": DATA_DIR / "dataset_200_objects.json",
            "ilp_key": None,
            "paper_claims": {
                "gap_vs_greedy": 1.67,
                "win_rate": 100.0,
            },
        },
    ]

    all_ok = True
    for c in checks:
        print(f"=== {c['label']} ===")
        if not c["model"].exists():
            print(f"  SKIP: checkpoint not found at {c['model']}\n")
            continue

        result = evaluate_checkpoint(c["model"], c["dataset"], device, c["ilp_key"])

        print(f"  Scenarios:       {result['n_scenarios']}")
        print(f"  GNN mean cost:   {result['mean_gnn_cost']:.1f}")
        print(f"  Greedy mean:     {result['mean_greedy_cost']:.1f}")
        print(f"  Gap vs greedy:   {result['gap_vs_greedy_pct']:.2f}%  (paper: {c['paper_claims']['gap_vs_greedy']:.2f}%)")
        print(f"  Gap std:         {result['gap_std']:.2f}%")
        print(f"  Win-rate:        {result['win_rate_pct']:.1f}%  (paper: {c['paper_claims']['win_rate']:.1f}%)")

        if result["mean_ilp_gap_pct"] is not None:
            paper_ilp = c["paper_claims"].get("ilp_gap")
            print(f"  ILP gap (GNN-ILP)/ILP: {result['mean_ilp_gap_pct']:.2f}%  (paper: {paper_ilp:.2f}%)")

        # Check tolerances
        claims = c["paper_claims"]
        tol = 1.0  # 1 percentage point tolerance for gap, 5% for win-rate
        gap_diff = abs(result["gap_vs_greedy_pct"] - claims["gap_vs_greedy"])
        wr_diff = abs(result["win_rate_pct"] - claims["win_rate"])
        if gap_diff > tol:
            print(f"  *** GAP MISMATCH: diff={gap_diff:.2f}pp ***")
            all_ok = False
        if wr_diff > 5.0:
            print(f"  *** WIN-RATE MISMATCH: diff={wr_diff:.1f}pp ***")
            all_ok = False
        if result["mean_ilp_gap_pct"] is not None and "ilp_gap" in claims:
            ilp_diff = abs(result["mean_ilp_gap_pct"] - claims["ilp_gap"])
            if ilp_diff > tol:
                print(f"  *** ILP GAP MISMATCH: diff={ilp_diff:.2f}pp ***")
                all_ok = False

        print()

    print("=" * 40)
    print("OVERALL:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
