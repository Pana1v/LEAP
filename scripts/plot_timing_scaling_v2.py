"""
Generate updated timing scaling figure showing all solver tiers.

Shows: Greedy, GNN standalone, GNN+ILP (k=15), Circuit cold, CBC
across problem sizes on a log-scale.

Output: references/figures/paper/fig_timing_scaling_v2.pdf + .png

Run from repo root:
  python3 scripts/plot_timing_scaling_v2.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "references" / "figures" / "paper"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data from benchmark results (seconds)
    # These are from the CPU runs; will be regenerated on GPU
    sizes = [10, 40, 100, 200]

    # GNN standalone times removed — not shown in figure to avoid confusion
    gnn_times = {}

    # Prefer the post-MTZ Circuit rerun (Phase 2); fall back to legacy results_k_sensitivity.json.
    pruned_times = {}
    circuit_cold_times = {}
    circuit_v8_path = ROOT / "experiments" / "ilp_timing_circuit_v8.json"
    if circuit_v8_path.exists():
        with open(circuit_v8_path) as f:
            ct = json.load(f)
        for n_str, e in ct["per_n"].items():
            n = int(n_str)
            circuit_cold_times[n] = e["unpruned"]["timing"]["mean_ms"] / 1000.0
            if e.get("leap"):
                pruned_times[n] = e["leap"]["timing"]["mean_ms"] / 1000.0
    else:
        k_sens_path = ROOT / "results_k_sensitivity.json"
        if k_sens_path.exists():
            with open(k_sens_path) as f:
                ks = json.load(f)
            for label, res in ks.items():
                n = res["N"]
                if "k_results" in res and "15" in res["k_results"]:
                    pruned_times[n] = res["k_results"]["15"]["mean_time"]
                if "cold_circuit" in res:
                    circuit_cold_times[n] = res["cold_circuit"]["mean_time"]

    # LKH (elkai, runs=1) standard-library heuristic
    lkh_times = {}
    lkh_path = ROOT / "experiments" / "lkh_tuned_v9.json"
    if lkh_path.exists():
        with open(lkh_path) as f:
            lj = json.load(f)["per_n"]
        for n_str, v in lj.items():
            lkh_times[int(n_str)] = v["mean_time_ms"] / 1000.0

    # Greedy is essentially instant
    greedy_times = {10: 0.0001, 40: 0.0005, 100: 0.002, 200: 0.005}

    # CBC times (approximate from benchmarks)
    cbc_times = {10: 0.5, 40: 60.0}  # N>=100 intractable

    fig, ax = plt.subplots(figsize=(5.5, 4))

    def plot_series(data, label, marker, color, linestyle="-", markersize=7):
        ns = sorted(data.keys())
        ts = [data[n] * 1000 for n in ns]  # convert to ms
        ax.plot(ns, ts, marker=marker, color=color, linewidth=1.8,
                linestyle=linestyle, label=label, markersize=markersize, zorder=3)

    plot_series(greedy_times, "Greedy NC", "s", "#ff7f0e", "--")
    if gnn_times:
        plot_series(gnn_times, "GNN standalone", "o", "#1f77b4")
    if pruned_times:
        plot_series(pruned_times, "LEAP (k=15)", "D", "#2ca02c")
    if lkh_times:
        plot_series(lkh_times, "LKH (elkai, runs=1)", "P", "#8c564b", "--")
    if circuit_cold_times:
        plot_series(circuit_cold_times, "Circuit (unpruned)", "^", "#9467bd", "--")
    if cbc_times:
        plot_series(cbc_times, "CBC (baseline ILP)", "v", "#d62728", ":")

    # Add "intractable" zone for CBC
    ax.axhspan(60000, 1e7, alpha=0.08, color="red")
    ax.text(150, 200000, "CBC\nintractable", fontsize=8, color="#d62728",
            ha="center", style="italic", alpha=0.7)

    ax.set_xlabel("Number of Objects ($N$)", fontsize=10)
    ax.set_ylabel("Time per scenario (ms)", fontsize=10)
    ax.set_yscale("log")
    ax.set_title("Solve Time Scaling by Method", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(5, 210)
    ax.set_ylim(0.05, 5e5)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_timing_scaling_v2.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_timing_scaling_v2.png", dpi=200, bbox_inches="tight")
    print(f"Saved to {OUT_DIR / 'fig_timing_scaling_v2.pdf'}")
    plt.close()


if __name__ == "__main__":
    main()
