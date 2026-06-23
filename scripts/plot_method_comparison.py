"""
Generate grouped bar chart comparing all methods across problem sizes.

Reads results_metaheuristic_n{10,40,200}.json and produces a grouped bar
chart of gap-vs-greedy for each method and problem size.

Output: references/figures/paper/fig_method_comparison.pdf + .png

Run from repo root:
  python3 scripts/plot_method_comparison.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "references" / "figures" / "paper"


def load_results(n):
    path = ROOT / f"results_metaheuristic_n{n}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sizes = [10, 40, 200]
    all_data = {}
    for n in sizes:
        r = load_results(n)
        if r:
            all_data[n] = r

    if not all_data:
        print("No results files found. Run benchmark_metaheuristics.py first.")
        return

    # Methods to show (in display order)
    method_map = [
        ("GLS_1000ms", "GLS"),
        ("SA_1000ms", "SA"),
        ("Tabu_1000ms", "Tabu"),
        ("GreedyDescent_1000ms", "Greedy\nDescent"),
        ("2opt_50restarts", "2-opt\n(50x)"),
    ]

    # Also add 2opt_10restarts as fallback for N=200
    method_keys_fallback = {"2opt_50restarts": "2opt_10restarts"}

    fig, ax = plt.subplots(figsize=(7, 3.5))

    n_methods = len(method_map)
    n_sizes = len(sizes)
    bar_width = 0.22
    group_gap = 0.15

    colors = {10: "#4e79a7", 40: "#59a14f", 200: "#e15759"}

    for si, n in enumerate(sizes):
        if n not in all_data:
            continue
        methods = all_data[n]["methods"]
        x_positions = []
        gaps = []
        for mi, (key, label) in enumerate(method_map):
            actual_key = key
            if key not in methods and key in method_keys_fallback:
                actual_key = method_keys_fallback[key]
            if actual_key in methods:
                gaps.append(methods[actual_key].get("gap_vs_greedy", 0))
            else:
                gaps.append(0)
            x_positions.append(mi * (n_sizes * bar_width + group_gap) + si * bar_width)

        bars = ax.bar(x_positions, gaps, bar_width, label=f"N={n}",
                      color=colors[n], edgecolor="white", linewidth=0.5)

        for bar, gap in zip(bars, gaps):
            if gap > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{gap:.1f}", ha="center", va="bottom", fontsize=6.5)

    # x-axis labels
    label_positions = [mi * (n_sizes * bar_width + group_gap) + (n_sizes - 1) * bar_width / 2
                       for mi in range(n_methods)]
    ax.set_xticks(label_positions)
    ax.set_xticklabels([label for _, label in method_map], fontsize=8)

    ax.set_ylabel("Improvement over Greedy NC (%)", fontsize=9)
    ax.set_title("Method Comparison Across Problem Sizes", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(8.5, ax.get_ylim()[1]))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_method_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_method_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved to {OUT_DIR / 'fig_method_comparison.pdf'}")
    plt.close()


if __name__ == "__main__":
    main()
