"""
LEAP (ours) vs LKH (elkai, standard lib) vs OR-Tools (GLS metaheuristic +
CP-SAT exact) — cost improvement and time, from the verified result JSONs.

Outputs to docs/figures/comparison/:
  cmp_time_scaling.png      time vs N (log)
  cmp_pareto_n200.png       cost improvement vs time at N=200
  cmp_cost_improvement.png  grouped bars, improvement vs greedy

Run from repo root:
  python3 scripts/plot_leap_lkh_ortools.py
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

# --- pull times/gaps from the verified JSONs ---------------------------------
ilp = json.load(open(EXP / "ilp_timing_circuit_v8.json"))["per_n"]
lkhj = json.load(open(EXP / "lkh_tuned_v9.json"))["per_n"]   # tuned: elkai runs=1
gls = json.load(open(EXP / "gls_circuit_v8.json"))["per_n"]

# time per scenario in seconds
t_leap = {n: ilp[str(n)]["leap"]["timing"]["mean_ms"] / 1000 for n in NS}
t_cpsat = {n: ilp[str(n)]["unpruned"]["timing"]["mean_ms"] / 1000 for n in NS}
t_lkh = {n: lkhj[str(n)]["mean_time_ms"] / 1000 for n in NS}
t_gls = {n: gls[str(n)]["time_mean_ms"] / 1000 for n in NS}

# cost improvement vs greedy (%)
g_lkh = {n: lkhj[str(n)]["gap_vs_greedy"] for n in NS}
g_gls = {n: gls[str(n)]["gap_vs_greedy_pct_mean"] for n in NS}
# LEAP (Table III) and CP-SAT optimum (documented constants)
g_leap = {10: 6.90, 40: 4.30, 200: 1.87}
g_cpsat = {10: 6.90, 40: 4.30, 200: 1.89}

METHODS = [
    ("LEAP (ours)",      "#2CA02C", "*", t_leap, g_leap),
    ("LKH (elkai)",      "#9467BD", "D", t_lkh,  g_lkh),
    ("OR-Tools GLS",     "#4878CF", "o", t_gls,  g_gls),
    ("OR-Tools CP-SAT",  "#6A6A6A", "s", t_cpsat, g_cpsat),
]


def _tfmt(x, _):
    if x < 1e-3:
        return f"{x*1e6:.0f}µs"
    if x < 1:
        return f"{x*1e3:.0f}ms"
    if x < 60:
        return f"{x:.1f}s"
    return f"{x/60:.1f}min"


def plot_time_scaling():
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color, mk, t, _ in METHODS:
        ax.plot(NS, [t[n] for n in NS], marker=mk, color=color, lw=1.8,
                ms=9 if mk == "*" else 6, label=name, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(NS)
    ax.set_xlabel("Number of objects $N$")
    ax.set_ylabel("Time per scenario (log)")
    ax.yaxis.set_major_formatter(FuncFormatter(_tfmt))
    ax.set_title("Solve time vs. problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    # annotate the N=200 spread
    for name, color, mk, t, _ in METHODS:
        ax.annotate(_tfmt(t[200], None), (200, t[200]), textcoords="offset points",
                    xytext=(6, 0), fontsize=7, color=color, va="center")
    ax.set_xlim(0, 235)
    fig.tight_layout()
    fig.savefig(OUT / "cmp_time_scaling.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "cmp_time_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pareto_n200():
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.set_yscale("log")
    ax.set_ylim(0.18, 70)
    offs = {"LEAP (ours)": (10, -2, "left"), "LKH (elkai)": (-12, -20, "right"),
            "OR-Tools GLS": (10, 2, "left"), "OR-Tools CP-SAT": (10, 2, "left")}
    for name, color, mk, t, g in METHODS:
        ax.scatter(g[200], t[200], color=color, marker=mk,
                   s=260 if mk == "*" else 95, edgecolors="k", linewidths=0.5,
                   zorder=4, label=name)
        dx, dy, ha = offs[name]
        ax.annotate(f"{name}\n({_tfmt(t[200], None)}, {g[200]:.2f}%)",
                    (g[200], t[200]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.5, ha=ha,
                    fontweight="bold" if "LEAP" in name else "normal")
    ax.yaxis.set_major_formatter(FuncFormatter(_tfmt))
    ax.set_xlabel("Cost improvement over Greedy NC (%)  → better")
    ax.set_ylabel("Time per scenario (log)  ← better")
    ax.set_title("Cost vs. time at $N=200$")
    ax.axvline(1.89, ls="--", color="0.6", lw=1)
    ax.text(1.89, 0.19, "optimum ", fontsize=7, color="0.4", va="bottom", ha="right")
    ax.text(0.02, 0.06, "↙ ideal (near-optimal & fast)", transform=ax.transAxes,
            fontsize=8, style="italic", color="#2CA02C")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1.78, 1.93)
    fig.tight_layout()
    fig.savefig(OUT / "cmp_pareto_n200.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "cmp_pareto_n200.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cost_improvement():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.2
    x = np.arange(len(NS))
    opt = {10: 6.90, 40: 4.30, 200: 1.89}
    for i, (name, color, mk, _, g) in enumerate(METHODS):
        ax.bar(x + (i - 1.5) * w, [g[n] for n in NS], w, color=color,
               edgecolor="k", linewidth=0.4, label=name)
    for j, n in enumerate(NS):
        ax.hlines(opt[n], x[j] - 2 * w, x[j] + 2 * w, color="red", ls="--", lw=1.2,
                  zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in NS])
    ax.set_ylabel("Cost improvement over Greedy NC (%)")
    ax.set_title("Cost parity: all reach ≈ optimum (red dash)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cmp_cost_improvement.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "cmp_cost_improvement.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_time_scaling()
    plot_pareto_n200()
    plot_cost_improvement()
    print("saved 3 figures to", OUT)
    for n in NS:
        print(f"N={n}: LEAP {g_leap[n]:.2f}%/{_tfmt(t_leap[n],0)}  "
              f"LKH {g_lkh[n]:.2f}%/{_tfmt(t_lkh[n],0)}  "
              f"GLS {g_gls[n]:.2f}%/{_tfmt(t_gls[n],0)}  "
              f"CP-SAT {g_cpsat[n]:.2f}%/{_tfmt(t_cpsat[n],0)}")


if __name__ == "__main__":
    main()
