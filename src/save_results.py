"""
Post-training documentation script.

Run after training completes:
    cd src
    python save_results.py --model ../models/gnn_200obj_<timestamp>.pt

Saves to docs/results/<timestamp>/:
  - eval_results.json          per-dataset metrics (GNN vs Greedy vs Random)
  - eval_summary.csv           one row per dataset
  - fig_learning_curves.pdf/png
  - fig_ilp_gap.pdf/png
  - fig_win_rate.pdf/png
  - training_log.csv           copy of the training log
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gnn_train import (
    GNNPolicy,
    rollout_model,
    load_scenarios,
    prepare_scenario,
    FEATURE_DIM,
)
from generate_paper_plots import (
    plot_learning_curves,
    plot_ilp_gap,
    plot_win_rate,
    load_training_log,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [
    ("200 objects", PROJECT_ROOT / "data/dataset_200_objects.json"),
    ("40 objects",  PROJECT_ROOT / "data/dataset_40_objects.json"),
    ("20 objects",  PROJECT_ROOT / "data/dataset_20_objects.json"),
    ("10 objects",  PROJECT_ROOT / "data/dataset_10_objects.json"),
]


def load_model(model_path: Path, device: torch.device) -> GNNPolicy:
    ck = torch.load(model_path, map_location=device)
    model = GNNPolicy(
        input_dim=FEATURE_DIM,
        hidden_dim=ck["hidden_dim"],
        heads=ck["heads"],
        dropout=ck["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def evaluate_dataset(model, dataset_path: Path, device, n=None):
    raw = load_scenarios(dataset_path)
    if n:
        raw = raw[:n]
    scenarios = [prepare_scenario(s, device) for s in raw]

    gnn_costs, greedy_costs, random_costs = [], [], []
    for s, r in zip(scenarios, raw):
        gnn_costs.append(rollout_model(model, s, device))
        greedy_costs.append(r["greedy_cost"])
        order = np.random.permutation(len(r["objects"]))
        cur = np.array(r["start"])
        cost = 0.0
        for i in order:
            obj = np.array(r["objects"][i])
            bin_pos = np.array(r["bins"][r["types"][i]])
            cost += np.linalg.norm(cur - obj) + np.linalg.norm(obj - bin_pos)
            cur = bin_pos
        random_costs.append(cost)

    gnn = np.array(gnn_costs)
    gr = np.array(greedy_costs)
    rnd = np.array(random_costs)
    gap_vs_greedy = (gr - gnn) / gr * 100
    gap_vs_random = (rnd - gnn) / rnd * 100
    wins = int((gnn < gr).sum())
    return {
        "n_scenarios": len(raw),
        "gnn_mean": float(gnn.mean()),
        "gnn_std": float(gnn.std()),
        "greedy_mean": float(gr.mean()),
        "greedy_std": float(gr.std()),
        "random_mean": float(rnd.mean()),
        "gap_vs_greedy_mean": float(gap_vs_greedy.mean()),
        "gap_vs_greedy_std": float(gap_vs_greedy.std()),
        "gap_vs_random_mean": float(gap_vs_random.mean()),
        "win_rate": float(wins / len(raw) * 100),
        "wins": wins,
    }


def plot_comparison_bar(results: dict, out_dir: Path):
    labels = list(results.keys())
    gnn_means = [results[l]["gnn_mean"] for l in labels]
    greedy_means = [results[l]["greedy_mean"] for l in labels]
    random_means = [results[l]["random_mean"] for l in labels]
    gaps = [results[l]["gap_vs_greedy_mean"] for l in labels]
    win_rates = [results[l]["win_rate"] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(labels))
    w = 0.25
    axes[0].bar(x - w, gnn_means, w, label="GNN", color="C0")
    axes[0].bar(x,     greedy_means, w, label="Greedy", color="C1")
    axes[0].bar(x + w, random_means, w, label="Random", color="C2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean Travel Cost")
    axes[0].set_title("Mean Cost Comparison")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    colors = ["C0" if g > 0 else "C3" for g in gaps]
    axes[1].bar(labels, gaps, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Gap vs Greedy (%)")
    axes[1].set_title("GNN Gap vs Greedy\n(positive = GNN wins)")
    axes[1].tick_params(axis="x", rotation=15, labelsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(labels, win_rates, color="C0")
    axes[2].set_ylim([0, 100])
    axes[2].set_ylabel("Win Rate (%)")
    axes[2].set_title("GNN Win Rate vs Greedy")
    axes[2].tick_params(axis="x", rotation=15, labelsize=9)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_comparison.pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(out_dir / "fig_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved fig_comparison.pdf/png")


def find_training_log(model_path: Path) -> Path | None:
    """Find the training_log.csv that matches this model's timestamp."""
    logs_dir = model_path.parent.parent / "src" / "logs"
    if not logs_dir.exists():
        logs_dir = Path("logs")
    candidates = sorted(logs_dir.glob("gnn_dataset_200_objects_*"), reverse=True)
    return candidates[0] / "training_log.csv" if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--log",   default=None,  help="Path to training_log.csv (auto-detected if omitted)")
    parser.add_argument("--out",   default=None,  help="Output dir (default: docs/results/<timestamp>)")
    parser.add_argument("--n",     type=int, default=500, help="Scenarios per dataset (default 500)")
    args = parser.parse_args()

    model_path = Path(args.model)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "docs" / "results" / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"Loading model: {model_path.name}")
    model = load_model(model_path, device)

    # Evaluate on all datasets
    print(f"\nEvaluating on {args.n} scenarios per dataset...")
    results = {}
    for label, ds_path in DATASETS:
        if not ds_path.exists():
            print(f"  skip {label} (not found)")
            continue
        print(f"  {label}...", end=" ", flush=True)
        results[label] = evaluate_dataset(model, ds_path, device, n=args.n)
        r = results[label]
        print(f"gap={r['gap_vs_greedy_mean']:+.2f}%  wins={r['wins']}/{r['n_scenarios']}")

    # Save JSON + CSV
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump({"model": str(model_path), "results": results}, f, indent=2)

    import csv
    with open(out_dir / "eval_summary.csv", "w", newline="") as f:
        fields = ["dataset", "n_scenarios", "gnn_mean", "greedy_mean", "random_mean",
                  "gap_vs_greedy_mean", "gap_vs_greedy_std", "gap_vs_random_mean", "win_rate"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for label, r in results.items():
            w.writerow({"dataset": label, **r})
    print("✓ Saved eval_results.json and eval_summary.csv")

    # Comparison bar chart
    plot_comparison_bar(results, out_dir)

    # Training curve plots
    log_path = Path(args.log) if args.log else find_training_log(model_path)
    if log_path and log_path.exists():
        shutil.copy(log_path, out_dir / "training_log.csv")
        df = load_training_log(log_path)
        plot_learning_curves(df, out_dir)
        plot_ilp_gap(df, out_dir)
        plot_win_rate(df, out_dir)
        print("✓ Saved training curve plots")
    else:
        print("⚠ training_log.csv not found — skipping curve plots")

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
