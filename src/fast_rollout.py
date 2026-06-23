"""
Plot per-method runtime distributions at N in {10, 40, 200}.

Distributions are reconstructed: we have published per-method (mean, std)
in some files and (mean only) in others. We fit a log-normal whose first
two moments match the published mean and std; for methods with no published
std we use a documented coefficient-of-variation (CV) per method class.
This is illustrative, not measured.

Layout: 3 panels (one per N). In each panel each method is a horizontal
violin on a log-time axis, with the published mean overlaid as a tick.
This is readable across the 5+ orders-of-magnitude time span (Greedy NC
microseconds vs. CP-SAT seconds at N=200).

Output: references/figures/paper/fig_runtime_histogram.png

Run from repo root:
  python3 scripts/plot_runtime_histogram.py
"""

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "references" / "figures" / "paper"

# CV (std/mean) assumption for methods without published time-std.
DEFAULT_CV = {
    "metaheuristic_budgeted": 0.02,
    "metaheuristic_local":    0.20,
    "solver":                 0.30,
    "deterministic":          0.10,
}

PALETTE = {
    "Greedy NC":         "#E8792B",
    "2-opt (OR-Tools)":  "#5B9BD5",
    "GLS (1s)":          "#4878CF",
    "LKH (elkai)":       "#8c564b",
    "CP-SAT Circuit":    "#6a6a6a",
    "LEAP (k=15)":       "#2CA02C",
}

ORDER = [
    "Greedy NC",
    "2-opt (OR-Tools)",
    "GLS (1s)",
    "LKH (elkai)",
    "CP-SAT Circuit",
    "LEAP (k=15)",
]


def lognormal_samples(mean_ms: float, std_ms: float, n: int = 500, rng=None):
    rng = rng or np.random.default_rng(0)
    if mean_ms <= 0:
        return np.full(n, max(mean_ms, 1e-6))
    if std_ms <= 0:
        return np.full(n, mean_ms)
    var = std_ms ** 2
    sigma2 = math.log(1.0 + var / (mean_ms ** 2))
    mu = math.log(mean_ms) - 0.5 * sigma2
    return rng.lognormal(mu, math.sqrt(sigma2), size=n)


def load_data(N: int):
    out = {}
    timing = json.loads((ROOT / "experiments" / "timing_results.json").read_text())
    meta = json.loads((ROOT / f"results_metaheuristic_n{N}.json").read_text())
    ksens = json.loads((ROOT / "results_k_sensitivity.json").read_text())
    # Prefer post-MTZ Circuit rerun (Phase 2) for Circuit + LEAP rows
    v8_path = ROOT / "experiments" / "ilp_timing_circuit_v8.json"
    v8 = None
    if v8_path.exists():
        v8 = json.loads(v8_path.read_text()).get("per_n", {}).get(str(N))

    t_block = timing.get(str(N), {})
    if "greedy_nc" in t_block and t_block["greedy_nc"]:
        g = t_block["greedy_nc"]
        out["Greedy NC"] = (g["mean_ms"], g.get("std_ms"))
    elif "Greedy_NC" in meta["methods"]:
        m = meta["methods"]["Greedy_NC"]["mean_time_ms"] or 0.05
        out["Greedy NC"] = (m, m * DEFAULT_CV["deterministic"])

    if "GLS_1000ms" in meta["methods"]:
        m = meta["methods"]["GLS_1000ms"]["mean_time_ms"]
        out["GLS (1s)"] = (m, m * DEFAULT_CV["metaheuristic_budgeted"])

    o2 = json.loads((ROOT / "experiments" / "ortools_2opt_v9.json").read_text())["per_n"].get(str(N))
    if o2:
        m = o2["mean_time_ms"]
        out["2-opt (OR-Tools)"] = (m, m * DEFAULT_CV["metaheuristic_local"])

    lk = json.loads((ROOT / "experiments" / "lkh_tuned_v9.json").read_text())["per_n"].get(str(N))
    if lk:
        m = lk["mean_time_ms"]
        out["LKH (elkai)"] = (m, m * DEFAULT_CV["solver"])

    if v8 is not None:
        u = v8["unpruned"]["timing"]
        out["CP-SAT Circuit"] = (u["mean_ms"], u.get("std_ms") or u["mean_ms"] * DEFAULT_CV["solver"])
        if v8.get("leap"):
            l = v8["leap"]["timing"]
            out["LEAP (k=15)"] = (l["mean_ms"], l.get("std_ms") or l["mean_ms"] * DEFAULT_CV["solver"])
    else:
        n_key = f"N={N}"
        if n_key in ksens:
            cold = ksens[n_key]["cold_circuit"]
            out["CP-SAT Circuit"] = (cold["mean_time"] * 1000.0,
                                     cold["mean_time"] * 1000.0 * DEFAULT_CV["solver"])
            if "15" in ksens[n_key]["k_results"]:
                leap = ksens[n_key]["k_results"]["15"]
                out["LEAP (k=15)"] = (leap["mean_time"] * 1000.0,
                                      leap["mean_time"] * 1000.0 * DEFAULT_CV["solver"])

    return out


def _time_fmt(x, _):
    if x < 1:
        return f"{x*1000:.0f}µs"
    if x < 1000:
        return f"{x:.0f} ms"
    if x < 60000:
        return f"{x/1000:.1f} s"
    return f"{x/60000:.1f} min"


def _draw_panel(ax, data, title, x_min, x_max):
    methods = [m for m in ORDER if m in data]
    rng = np.random.default_rng(42)

    positions = list(range(len(methods)))
    sample_log = []
    