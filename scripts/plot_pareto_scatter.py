"""
Generate cost-vs-time scatter for N=200 (single figure).

x-axis: Mean route cost
y-axis: Compute time (log scale, tight range)

Output: references/figures/paper/fig_pareto_n200.png

Run from repo root:
  python3 scripts/plot_pareto_scatter.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "references" / "figures" / "paper"

STYLE = {
    "greedy": dict(color="#E8792B", marker="^", zorder=3, s=40),
    "meta":   dict(color="#4878CF", marker="o", zorder=4, s=40),
    "lkh":    dict(color="#8c564b", marker="P", zorder=4, s=70),
    "exact":  dict(color="#6a6a6a", marker="s", zorder=5, s=40),
    "ours":   dict(color="#2CA02C", marker="*", zorder=6, s=160),
}

GREEDY_200 = 18781.0
OPT_200 = GREEDY_200 * (1 - 0.0189)  # verified unpruned CP-SAT optimum gap (Table 1)

DATA = [
    ("Greedy NC",       "greedy", GREEDY_200,            1.0,      8, 0),
    ("GLS (1s)",        "meta",   GREEDY_200 * 0.9812,   1000,     8, 0),
    ("LKH (elkai)",     "lkh",    OPT_200,               5179,     8, 14),
    ("CP-SAT Circuit",  "exact",  OPT_200,               4036,     8, -16),
    ("LEAP (k=15)",     "ours",   OPT_200 * 1.0002,      461,      8, -5),
]


def _time_fmt(x, _):
    if x < 1:
        return f"{x*1000:.0f}\u00b5s"
    elif x < 1000:
        return f"{x:.0f}ms"
    elif x < 60000:
        return f"{x/1000:.1f}s"
    else:
        return f"{x/60000:.1f}min"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.3, 3.8))

    # Use a custom log scale with limited range to spread the points
    ax.set_yscale("log")
    ax.set_ylim(0.3, 600000)  # 0.3ms to 10min — tighter than before

    for label, cat, cost, time_ms, dx, dy in DATA:
        sty = STYLE[cat]
        ax.scatter(cost, time_ms, **sty, edgecolors="k", linewidths=0.4)
        fw = "bold" if cat == "ours" else "normal"
        fs = 7.5 if cat == "ours" else 6.5
        ha = "left" if dx > 0 else "right"
        ax.annotate(label, (cost, time_ms), textcoords="offset points",
                    xytext=(dx, dy), fontsize=fs, fontweight=fw, ha=ha,
                    va="center")

    # Add timing annotations for key points
    time_labels = {
        "LEAP (k=15)": (-8, -12),
        "CP-SAT Circuit": (8, -10),
        "LKH (elkai)": (8, 6),
        "GLS": (8, -10),
    }
    for label, cat, cost, time_ms, dx, dy in DATA:
        if label in time_labels:
            tdx, tdy = time_labels[label]
            t_str = _time_fmt(time_ms, None)
            tha = "left" if tdx > 0 else "right"
            ax.annotate(f"({t_str})", (cost, time_ms),
                        textcoords="offset points", xytext=(tdx, tdy),
                        fontsize=5.5, color="0.4", ha=tha, va="center")

    # x-axis
    costs = [c for _, _, c, _, _, _ in DATA]
    span = max(costs) - min(costs)
    ax.set_xlim(min(costs) - span * 0.06, max(costs) + span * 0.12)

    # ideal corner
    ax.text(0.03, 0.03, "\u2190 Ideal (low cost, fast)",
            transform=ax.transAxes, fontsize=6, fontstyle="italic",
            color="#2CA02C", alpha=0.7)

    # draw horizontal dashed lines from LEAP and CP-SAT Circuit to show time gap
    leap_t = 461
    cpsat_t = 4036
    ax.annotate("", xy=(18400, leap_t), xytext=(18400, cpsat_t),
                arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8))
    ax.text(18350, (leap_t * cpsat_t)**0.5, "8.75\u00d7",
            fontsize=6.5, color="0.4", va="center", ha="right",
            fontweight="bold")

    # Cost-axis gap from Greedy NC -> LEAP (1.9% improvement at N=200)
    leap_cost = OPT_200 * 1.0002
    greedy_cost = GREEDY_200
    ax.annotate("", xy=(leap_cost, 1.0), xytext=(greedy_cost, 1.0),
                arrowprops=dict(arrowstyle="<->", color="#E8792B", lw=0.9,
                                alpha=0.85))
    ax.text((leap_cost + greedy_cost) / 2.0, 0.55,
            "\u20131.9% cost\n(Greedy NC \u2192 LEAP)",
            fontsize=5.8, color="#A04A0F", ha="center", va="center",
            fontweight="bold")

    ax.set_xlabel("Mean route cost ($N = 200$)", fontsize=8.5)
    ax.set_ylabel("Compute time per scenario", fontsize=8.5)
    ax.set_title("Cost vs. Compute Time ($N = 200$)", fontsize=7,
                 fontweight="bold", pad=6)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(_time_fmt))

    handles = [
        Line2D([], [], color="#E8792B", marker="^", ls="", ms=5,
               mec="k", mew=0.4, label="Greedy NC"),
        Line2D([], [], color="#4878CF", marker="o", ls="", ms=5,
               mec="k", mew=0.4, label="Metaheuristic"),
        Line2D([], [], color="#8c564b", marker="P", ls="", ms=6,
               mec="k", mew=0.4, label="LKH (heuristic)"),
        Line2D([], [], color="#6a6a6a", marker="s", ls="", ms=5,
               mec="k", mew=0.4, label="Exact solver"),
        Line2D([], [], color="#2CA02C", marker="*", ls="", ms=8,
               mec="k", mew=0.4, label="LEAP (ours)"),
    ]
    ax.legend(handles=handles, fontsize=5.5, loc="center right",
              framealpha=0.9, edgecolor="0.8")

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_pareto_n200.{ext}", dpi=200,
                    bbox_inches="tight", pad_inches=0.05)
    print("Saved fig_pareto_n200")
    plt.close(fig)


if __name__ == "__main__":
    main()
