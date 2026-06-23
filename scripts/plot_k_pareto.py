"""
Generate k-sensitivity Pareto figure for the paper.

Reads results_k_sensitivity.json and produces a dual-axis plot:
  x-axis: k (neighbourhood size)
  left y-axis: solve time (s)
  right y-axis: max optimality gap (%)

Output: references/figures/paper/fig_k_pareto.pdf + .png

Run from repo root:
  python3 scripts/plot_k_pareto.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
# Prefer the fast-rollout k-sweep if available; fall back to the legacy JSON.
_FAST = REPO / "experiments" / "k_sensitivity_fast_v8.json"
RESULTS_PATH = _FAST if _FAST.exists() else REPO / "results_k_sensitivity.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "references" / "figures" / "paper"


def main():
    if not RESULTS_PATH.exists():
        print(f"Missing {RESULTS_PATH} — run benchmark_k_sensitivity.py first")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        raw = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
    ax2 = ax1.twinx()

    markers = {"N=40": "o", "N=100": "s", "N=200": "D"}
    colors_time = {"N=40": "#1f77b4", "N=100": "#2ca02c", "N=200": "#d62728"}
    colors_gap = {"N=40": "#aec7e8", "N=100": "#98df8a", "N=200": "#ff9896"}

    for label, res in raw.items():
        k_results = res.get("k_results", {})
        if not k_results:
            continue
        ks = sorted(k_results.keys(), key=int)
        k_vals = [int(k) for k in ks]
        times = [k_results[k]["mean_time"] for k in ks]
        max_gaps = [k_results[k]["max_gap_vs_cold_pct"] for k in ks]

        ax1.plot(k_vals, times, marker=markers.get(label, "o"),
                 color=colors_time[label], linewidth=1.5,
                 label=f"{label} time", zorder=3)
        ax2.plot(k_vals, max_gaps, marker=markers.get(label, "o"),
                 color=colors_gap[label], linewidth=1.5, linestyle="--",
                 label=f"{label} max gap", zorder=2)

    ax1.set_xlabel("Neighbourhood size $k$", fontsize=10)
    ax1.set_ylabel("Solve time (s)", fontsize=10, color="#333333")
    ax2.set_ylabel("Max optimality gap (%)", fontsize=10, color="#333333")

    ax1.set_xticks([3, 5, 10, 15, 20, 30, 50])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_xaxis().set_tick_params(which="minor", size=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7, loc="upper right", framealpha=0.9)

    ax1.grid(True, alpha=0.3, which="both")
    plt.title("$k$-Sensitivity: Quality–Speed Pareto Frontier", fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()

    fig.savefig(OUT_DIR / "fig_k_pareto.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_k_pareto.png", dpi=200, bbox_inches="tight")
    print(f"Saved to {OUT_DIR / 'fig_k_pareto.pdf'}")
    plt.close()


if __name__ == "__main__":
    main()
