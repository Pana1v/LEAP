"""
Quick comparison of all methods AFTER LKH tuning (elkai runs=1).
Left: time per scenario vs N (log). Right: cost improvement vs Greedy at N=200
with Greedy in the mix (full scale => optimizer differences are parity).

All numbers from the verified result JSONs. LEAP is ours; everything else is a
standard library. Output: docs/figures/comparison/cmp_after_lkh_tuning.png
Run from repo root: python3 scripts/plot_methods_after_lkh_tuning.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "figures" / "comparison"
NS = [10, 40, 200]

ilp = json.load(open(EXP / "ilp_timing_circuit_v8.json"))["per_n"]
gls = json.load(open(EXP / "gls_circuit_v8.json"))["per_n"]
o2 = json.load(open(EXP / "ortools_2opt_v9.json"))["per_n"]
lkh = json.load(open(EXP / "lkh_tuned_v9.json"))["per_n"]

def s(d, key):  # ms -> s
    return {n: d[str(n)][key] / 1000.0 for n in NS}

t = {
    "Greedy NC":        {n: 0.0005 for n in NS},
    "2-opt (OR-Tools)": s(o2, "mean_time_ms"),
    "GLS (OR-Tools)":   s(gls, "time_mean_ms"),
    "LEAP (ours)":      {n: ilp[str(n)]["leap"]["timing"]["mean_ms"] / 1000.0 for n in NS},
    "CP-SAT (exact)":   {n: ilp[str(n)]["unpruned"]["timing"]["mean_ms"] / 1000.0 for n in NS},
    "LKH (elkai, runs=1)": s(lkh, "mean_time_ms"),
}
gap200 = {
    "Greedy NC": 0.0,
    "2-opt (OR-Tools)": o2["200"]["gap_vs_greedy"],
    "GLS (OR-Tools)": gls["200"]["gap_vs_greedy_pct_mean"],
    "LEAP (ours)": 1.87,
    "CP-SAT (exact)": 1.89,
    "LKH (elkai, runs=1)": lkh["200"]["gap_vs_greedy"],
}
COL = {
    "Greedy NC": "#E8792B", "2-opt (OR-Tools)": "#5B9BD5", "GLS (OR-Tools)": "#4878CF",
    "LEAP (ours)": "#2CA02C", "CP-SAT (exact)": "#6A6A6A", "LKH (elkai, runs=1)": "#8C564B",
}
MK = {
    "Greedy NC": "v", "2-opt (OR-Tools)": "s", "GLS (OR-Tools)": "o",
    "LEAP (ours)": "*", "CP-SAT (exact)": "^", "LKH (elkai, runs=1)": "P",
}


def tfmt(x, _):
    if x < 1e-3: return f"{x*1e6:.0f}µs"
    if x < 1: return f"{x*1e3:.0f}ms"
    if x < 60: return f"{x:.1f}s"
    return f"{x/60:.0f}min"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: time vs N
    for name, tm in t.items():
        axL.plot(NS, [tm[n] for n in NS], marker=MK[name], color=COL[name],
                 lw=1.8, ms=11 if name == "LEAP (ours)" else 6, label=name, zorder=3)
    axL.set_yscale("log"); axL.set_xscale("log"); axL.set_xticks(NS)
    axL.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axL.yaxis.set_major_formatter(FuncFormatter(tfmt))
    axL.set_xlabel("Number of objects $N$"); axL.set_ylabel("Time per scenario (log)")
    axL.set_title("Solve time vs. $N$ (after LKH tuning)")
    axL.grid(True, which="both", alpha=0.3); axL.legend(fontsize=8)
    for name in ("LKH (elkai, runs=1)", "LEAP (ours)", "CP-SAT (exact)"):
        axL.annotate(tfmt(t[name][200], 0), (200, t[name][200]), textcoords="offset points",
                     xytext=(6, 0), fontsize=7, color=COL[name], va="center")

    # Right: cost improvement at N=200, greedy in the mix
    names = ["Greedy NC", "2-opt (OR-Tools)", "GLS (OR-Tools)", "LKH (elkai, runs=1)",
             "CP-SAT (exact)", "LEAP (ours)"]
    vals = [gap200[n] for n in names]
    axR.bar(range(len(names)), vals, color=[COL[n] for n in names], edgecolor="k", linewidth=0.4)
    axR.axhline(1.89, ls="--", color="red", lw=1.2)
    axR.text(0.1, 1.89, " optimum", color="red", fontsize=8, va="bottom")
    for i, v in enumerate(vals):
        axR.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
    axR.set_xticks(range(len(names)))
    axR.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
    axR.set_ylabel("Cost improvement over Greedy NC (%)")
    axR.set_title("Cost at $N=200$ (greedy in the mix → parity)")
    axR.set_ylim(0, 2.15); axR.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "cmp_after_lkh_tuning.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "cmp_after_lkh_tuning.pdf", bbox_inches="tight")
    print("saved", OUT / "cmp_after_lkh_tuning.png")
    for n in NS:
        print(f"N={n}: " + "  ".join(f"{k.split(' (')[0]} {tfmt(v[n],0)}" for k, v in t.items()))


if __name__ == "__main__":
    main()
