"""
Anytime speed-to-quality: GLS gap-to-optimum vs wall-clock budget (a curve),
with LEAP as a single (time, gap) point. Shows the crossover:
LEAP wins at N=100, GLS wins at N=200. All numbers from the verified JSONs.
LEAP wall-clock uses the corrected ilp_timing; GLS curve + LEAP gap from the
budget sweep. Output: docs/figures/comparison/anytime_gls_vs_leap.png
Run from repo root: python3 scripts/plot_anytime_gls_vs_leap.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
g = json.load(open(ROOT / "experiments" / "gls_budget_sweep_v9.json"))["per_n"]
ilp = json.load(open(ROOT / "experiments" / "ilp_timing_circuit_v8.json"))["per_n"]
OUT = ROOT / "docs" / "figures" / "comparison"


def tfmt(x, _):
    return f"{x/1000:.0f}s" if x >= 1000 else f"{x:.0f}ms"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, N in zip(axes, ("100", "200")):
        sweep = g[N]["sweep"]
        bx = [s["budget_ms"] for s in sweep]
        by = [s["gls_gap_pct"] for s in sweep]
        leap_t = ilp[N]["leap"]["timing"]["mean_ms"]
        leap_gap = g[N]["leap_gap_pct"]

        ax.plot(bx, by, "-o", color="#4878CF", lw=2, ms=5, label="GLS (OR-Tools), anytime", zorder=3)
        ax.scatter([leap_t], [leap_gap], marker="*", s=320, color="#2CA02C",
                   edgecolors="k", linewidths=0.6, zorder=5, label="LEAP (ours), single run")
        # guide lines from LEAP point
        ax.axhline(leap_gap, ls=":", color="#2CA02C", lw=1, alpha=0.7)
        ax.axvline(leap_t, ls=":", color="#2CA02C", lw=1, alpha=0.7)
        ax.annotate(f"LEAP\n{tfmt(leap_t,0)}, {leap_gap:.3f}%", (leap_t, leap_gap),
                    textcoords="offset points", xytext=(10, -28), fontsize=8,
                    fontweight="bold", color="#1f6b1f")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(tfmt))
        ax.set_xlabel("Wall-clock budget per scenario (log)")
        ax.set_ylabel("Gap to optimum (%, log) — lower is better")
        winner = "LEAP wins (faster to its quality)" if N == "100" else "GLS wins (better at equal/less time)"
        ax.set_title(f"$N={N}$ — {winner}", fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper right")
        # shade the region where GLS beats LEAP (left of LEAP time AND below LEAP gap is GLS-better-on-both)
        ax.axvspan(ax.get_xlim()[0], leap_t, color="#4878CF", alpha=0.04)

    fig.suptitle("Anytime quality vs. wall-clock: off-the-shelf GLS curve vs. LEAP point  (lower-left = better)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "anytime_gls_vs_leap.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "anytime_gls_vs_leap.pdf", bbox_inches="tight")
    print("saved", OUT / "anytime_gls_vs_leap.png")
    for N in ("100", "200"):
        lt = ilp[N]["leap"]["timing"]["mean_ms"]; lg = g[N]["leap_gap_pct"]
        print(f"N={N}: LEAP {lt:.0f}ms @ {lg:.4f}% | GLS sweep " +
              " ".join(f"{s['budget_ms']}ms:{s['gls_gap_pct']:.4f}%" for s in g[N]["sweep"]))


if __name__ == "__main__":
    main()
