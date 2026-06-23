"""
Generate the TODO figures for the paper.
  Fig 1: Graph topology diagram (cyclic star)
  Fig 2: Cost comparison across problem sizes
  Fig 3: Box plot of per-scenario improvements
Run from src/:  python3 generate_paper_figures.py
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

from gnn_train import (
    GNNPolicy, FEATURE_DIM, DEFAULT_DROPOUT, WORKSPACE_SIZE,
    load_scenarios, split_dataset, prepare_scenario, rollout_model,
)

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "docs" / "figures" / "paper"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

BIN_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.relative_to(PROJECT_ROOT)}")


def load_model(path: Path, device: torch.device):
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
        hd = int(blob.get("hidden_dim", 128))
        heads = int(blob.get("heads", 4))
        dp = float(blob.get("dropout", DEFAULT_DROPOUT))
    else:
        sd = blob
        hd = sd["convs.0.bias"].shape[0] if "convs.0.bias" in sd else 128
        heads = sd["convs.0.att_src"].shape[1] if "convs.0.att_src" in sd else 4
        dp = DEFAULT_DROPOUT
    model = GNNPolicy(FEATURE_DIM, hd, heads, dp).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════
# FIG 1: GRAPH TOPOLOGY DIAGRAM
# ═══════════════════════════════════════════════════════

def fig_graph_topology(out: Path):
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.0, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Robot at center
    robot_xy = (0, 0)
    robot_box = FancyBboxPatch((-0.22, -0.22), 0.44, 0.44,
                                boxstyle="round,pad=0.06", facecolor="#34495e",
                                edgecolor="#2c3e50", linewidth=2.5, zorder=10)
    ax.add_patch(robot_box)
    ax.text(0, 0, "R", ha="center", va="center", fontsize=18,
            fontweight="bold", color="white", zorder=11)

    # Bins at corners — wider spacing
    bin_positions = [(-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)]
    bin_labels = ["$b_0$", "$b_1$", "$b_2$", "$b_3$"]
    for i, (bx, by) in enumerate(bin_positions):
        diamond = plt.Polygon(
            [(bx, by + 0.22), (bx + 0.22, by), (bx, by - 0.22), (bx - 0.22, by)],
            facecolor=BIN_COLORS[i], edgecolor="white",
            linewidth=2.5, zorder=10, alpha=0.9,
        )
        ax.add_patch(diamond)
        ax.text(bx, by, bin_labels[i], ha="center", va="center", fontsize=14,
                fontweight="bold", color="white", zorder=11)

    # 4 objects in a ring (reduced from 6 for clarity)
    n_obj = 4
    obj_types = [0, 1, 2, 3]
    obj_labels = ["$o_1$", "$o_2$", "$o_3$", "$o_4$"]
    obj_positions = []
    for i in range(n_obj):
        angle = 2 * np.pi * i / n_obj - np.pi / 2
        r = 0.72
        ox, oy = r * np.cos(angle), r * np.sin(angle)
        obj_positions.append((ox, oy))

    for i, (ox, oy) in enumerate(obj_positions):
        t = obj_types[i]
        circle = plt.Circle((ox, oy), 0.15, facecolor=BIN_COLORS[t],
                              edgecolor="white", linewidth=2, zorder=10, alpha=0.9)
        ax.add_patch(circle)
        ax.text(ox, oy, obj_labels[i], ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", zorder=11)

    # Highlighted cycle: o1 (type 0 → bin b0)
    highlight_idx = 0
    ox, oy = obj_positions[highlight_idx]
    t = obj_types[highlight_idx]
    bx, by = bin_positions[t]

    curve_kw = dict(arrowstyle="-|>", mutation_scale=20, linewidth=3.0)

    # Robot → Object (pick) — BLUE, curved
    ax.annotate("", xy=(ox, oy), xytext=robot_xy,
                arrowprops=dict(**curve_kw, color="#2980b9",
                                connectionstyle="arc3,rad=0.15",
                                shrinkA=16, shrinkB=12))
    # Object → Bin (place) — GREEN, curved
    ax.annotate("", xy=(bx, by), xytext=(ox, oy),
                arrowprops=dict(**curve_kw, color="#27ae60",
                                connectionstyle="arc3,rad=0.15",
                                shrinkA=12, shrinkB=16))
    # Bin → Robot (feedback, not a physical travel leg) — GREY dashed
    fb_kw = dict(arrowstyle="-|>", mutation_scale=18, linewidth=2.0,
                 linestyle="dashed", color="#7f8c8d", alpha=0.75)
    ax.annotate("", xy=robot_xy, xytext=(bx, by),
                arrowprops=dict(**fb_kw,
                                connectionstyle="arc3,rad=0.15",
                                shrinkA=16, shrinkB=16))

    # Faint dashed edges for other objects
    for i in range(1, n_obj):
        ox2, oy2 = obj_positions[i]
        t2 = obj_types[i]
        bx2, by2 = bin_positions[t2]
        faint_kw = dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.0,
                        alpha=0.45, color="#7f8c8d", linestyle=(0, (4, 2)),
                        connectionstyle="arc3,rad=0.1",
                        shrinkA=16, shrinkB=12)
        ax.annotate("", xy=(ox2, oy2), xytext=robot_xy, arrowprops=dict(**faint_kw))
        faint_kw["shrinkA"] = 12
        faint_kw["shrinkB"] = 16
        ax.annotate("", xy=(bx2, by2), xytext=(ox2, oy2), arrowprops=dict(**faint_kw))
        faint_kw["shrinkA"] = 16
        faint_kw["shrinkB"] = 16
        ax.annotate("", xy=robot_xy, xytext=(bx2, by2), arrowprops=dict(**faint_kw))

    # Edge labels on the highlighted cycle
    mid_ro = ((robot_xy[0] + ox) / 2 + 0.12, (robot_xy[1] + oy) / 2 - 0.15)
    ax.text(mid_ro[0], mid_ro[1], "pick", fontsize=11, color="#2980b9",
            fontstyle="italic", fontweight="bold", zorder=12)

    mid_ob = ((ox + bx) / 2 - 0.25, (oy + by) / 2 + 0.05)
    ax.text(mid_ob[0], mid_ob[1], "place", fontsize=11, color="#27ae60",
            fontstyle="italic", fontweight="bold", zorder=12)

    mid_br = ((bx + robot_xy[0]) / 2 - 0.05, (by + robot_xy[1]) / 2 + 0.18)
    ax.text(mid_br[0], mid_br[1], "feedback", fontsize=11, color="#7f8c8d",
            fontstyle="italic", fontweight="bold", zorder=12)

    # Legend — positioned to avoid overlapping b2/o3 nodes
    legend_els = [
        Line2D([0], [0], color="#2980b9", lw=3, label="Pick (cost leg): Robot $\\to$ Object"),
        Line2D([0], [0], color="#27ae60", lw=3, label="Place (cost leg): Object $\\to$ Bin"),
        Line2D([0], [0], color="#7f8c8d", lw=2, ls="--",
               label="Feedback (info only): Bin $\\to$ Robot"),
        Line2D([0], [0], color="#bdc3c7", lw=1, ls="--", alpha=0.6,
               label="Other candidate cycles"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#34495e",
               ms=14, label="Robot node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6",
               ms=11, label="Object nodes (coloured by type)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#95a5a6",
               ms=11, label="Bin nodes"),
    ]
    ax.legend(handles=legend_els, loc="upper left", framealpha=0.95,
              fontsize=10, edgecolor="#cccccc",
              bbox_to_anchor=(-0.02, 1.02), borderpad=0.4, labelspacing=0.3)

    ax.set_title("Cyclic Star Topology", fontsize=18,
                 fontweight="bold", pad=12)

    savefig(fig, out / "fig1_graph_topology.png")


# ═══════════════════════════════════════════════════════
# FIG 2: COST COMPARISON ACROSS PROBLEM SIZES
# ═══════════════════════════════════════════════════════

def evaluate_model_on_dataset(model_path, dataset_path, device, max_scenarios=200):
    scenarios_raw = load_scenarios(dataset_path)
    _, val_raw = split_dataset(scenarios_raw, val_split=0.1, seed=0)
    val_raw = val_raw[:max_scenarios]
    scenarios = [prepare_scenario(s, device) for s in val_raw]
    model = load_model(model_path, device)

    gnn_costs, greedy_costs = [], []
    with torch.no_grad():
        for s in scenarios:
            gnn_costs.append(rollout_model(model, s, device))
            greedy_costs.append(s["greedy_cost"])

    return {
        "gnn": np.array(gnn_costs),
        "greedy": np.array(greedy_costs),
        "n_scenarios": len(scenarios),
    }


def fig_cost_comparison(results_by_n, out: Path):
    ns = sorted(results_by_n.keys())

    # --- Grouped bar chart ---
    fig, ax = plt.subplots(figsize=(11, 6.5))

    x = np.arange(len(ns))
    width = 0.35
    gnn_means = [results_by_n[n]["gnn"].mean() for n in ns]
    gnn_stds = [results_by_n[n]["gnn"].std() for n in ns]
    greedy_means = [results_by_n[n]["greedy"].mean() for n in ns]
    greedy_stds = [results_by_n[n]["greedy"].std() for n in ns]

    bars_gnn = ax.bar(x - width/2, gnn_means, width, yerr=gnn_stds, capsize=3,
                       color="#2980b9", alpha=0.85, label="GNN (ours)", edgecolor="white")
    bars_greedy = ax.bar(x + width/2, greedy_means, width, yerr=greedy_stds, capsize=3,
                          color="#e74c3c", alpha=0.85, label="Greedy", edgecolor="white")

    ax.set_xlabel("Number of Objects ($N$)", fontsize=16)
    ax.set_ylabel("Mean Travel Cost", fontsize=16)
    ax.set_title("Cost Comparison Across Problem Sizes", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$N={n}$" for n in ns], fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(fontsize=14)

    # Add gap % annotations above GNN bars
    for i, n in enumerate(ns):
        gap = (greedy_means[i] - gnn_means[i]) / greedy_means[i] * 100
        y_pos = max(gnn_means[i], greedy_means[i]) + max(gnn_stds[i], greedy_stds[i]) + greedy_means[i] * 0.02
        ax.text(i - width / 2, gnn_means[i] + gnn_stds[i] + greedy_means[i] * 0.01,
                f"+{gap:.1f}%", ha="center", va="bottom", fontsize=14,
                fontweight="bold", color="#27ae60")

    savefig(fig, out / "fig2_cost_comparison.png")

    # --- Gap % bar chart (cleaner view) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    gaps = [(results_by_n[n]["greedy"].mean() - results_by_n[n]["gnn"].mean()) /
             results_by_n[n]["greedy"].mean() * 100 for n in ns]
    per_scenario_gaps = {n: (results_by_n[n]["greedy"] - results_by_n[n]["gnn"]) /
                           results_by_n[n]["greedy"] * 100 for n in ns}
    gap_stds = [per_scenario_gaps[n].std() for n in ns]

    bars = ax.bar(x, gaps, 0.5, yerr=gap_stds, capsize=5,
                   color=["#2980b9" if g > 0 else "#e74c3c" for g in gaps],
                   alpha=0.85, edgecolor="white")
    ax.axhline(y=0, color="#2c3e50", lw=1)
    ax.set_xlabel("Number of Objects ($N$)", fontsize=16)
    ax.set_ylabel("Improvement over Greedy (%)", fontsize=16)
    ax.set_title("GNN Improvement over Greedy by Problem Size",
                 fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$N={n}$" for n in ns], fontsize=14)
    ax.tick_params(axis="y", labelsize=13)

    for i, (g, s) in enumerate(zip(gaps, gap_stds)):
        ax.text(i, g + s + 0.2, f"{g:.2f}%", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#2980b9")

    savefig(fig, out / "fig2_gap_by_size.png")


# ═══════════════════════════════════════════════════════
# FIG 3: BOX PLOT OF IMPROVEMENTS
# ═══════════════════════════════════════════════════════

def fig_boxplot(results_by_n, out: Path):
    ns = sorted(results_by_n.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    # Cost difference: negative means GNN is better
    data = []
    labels = []
    for n in ns:
        diff = results_by_n[n]["gnn"] - results_by_n[n]["greedy"]
        data.append(diff)
        labels.append(f"$N={n}$")

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True,
                     flierprops=dict(marker="o", markersize=3, alpha=0.4),
                     widths=0.5)

    colors_box = ["#3498db", "#2980b9", "#2471a3"]
    # Extend colors if more problem sizes
    while len(colors_box) < len(ns):
        colors_box.append("#1a5276")

    for i, (patch, median) in enumerate(zip(bp["boxes"], bp["medians"])):
        patch.set_facecolor(colors_box[i % len(colors_box)])
        patch.set_alpha(0.7)
        median.set_color("#2c3e50")
        median.set_linewidth(2.5)

    ax.axhline(y=0, color="#e74c3c", ls="--", lw=1.5, label="Break-even (GNN = Greedy)")

    # Annotate median and win rate inside each box
    for i, n in enumerate(ns):
        diff = data[i]
        med = float(np.median(diff))
        win_pct = (diff < 0).sum() / len(diff) * 100
        ax.text(i + 1, med, f"med: {med:.0f}\nwin: {win_pct:.0f}%",
                fontsize=10, va="center", ha="center", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#cccccc", alpha=0.85))

    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Cost Difference (GNN $-$ Greedy)")
    ax.set_title("Distribution of Per-Scenario Cost Differences\n(Negative = GNN better)")
    ax.legend(loc="upper right")

    savefig(fig, out / "fig3_boxplot_improvements.png")

    # Also generate a gap% version (more intuitive)
    fig, ax = plt.subplots(figsize=(8, 6))
    data_gap = []
    for n in ns:
        gap = (results_by_n[n]["greedy"] - results_by_n[n]["gnn"]) / results_by_n[n]["greedy"] * 100
        data_gap.append(gap)

    bp = ax.boxplot(data_gap, tick_labels=labels, patch_artist=True, showfliers=True,
                     flierprops=dict(marker="o", markersize=3, alpha=0.4),
                     widths=0.5)

    for i, (patch, median) in enumerate(zip(bp["boxes"], bp["medians"])):
        patch.set_facecolor(colors_box[i % len(colors_box)])
        patch.set_alpha(0.7)
        median.set_color("#2c3e50")
        median.set_linewidth(2.5)

    ax.axhline(y=0, color="#e74c3c", ls="--", lw=1.5, label="Break-even")

    for i, n in enumerate(ns):
        gap = data_gap[i]
        med = float(np.median(gap))
        win_pct = (gap > 0).sum() / len(gap) * 100
        q3 = float(np.percentile(gap, 75))
        ax.text(i + 1, q3 + 0.8, f"med: {med:.1f}%\nwin: {win_pct:.0f}%",
                fontsize=11, va="bottom", ha="center", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#cccccc", alpha=0.85))

    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Improvement over Greedy (%)")
    ax.set_title("Distribution of Per-Scenario Improvements\n(Positive = GNN better)")
    ax.legend(loc="upper right")

    savefig(fig, out / "fig3_boxplot_gap_pct.png")


# ═══════════════════════════════════════════════════════
# FIG 4: ABLATION STUDY
# ═══════════════════════════════════════════════════════

def fig_ablations(out: Path):
    results_path = PROJECT_ROOT / "experiments" / "ablations" / "ablation_results_40obj.json"
    if not results_path.exists():
        print(f"  SKIP ablation figure: {results_path} not found")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Display order and labels (deduplicated — shared entries shown once)
    variants = [
        ("A1_cyclic_star",     "Cyclic Star\n(ours)"),
        ("A1_fully_connected", "Fully\nConnected"),
        ("A1_knn",             "k-NN\n(k=5)"),
        ("A2_gcn",             "GCN"),
        ("A2_mlp",             "MLP\n(no graph)"),
        ("A3_no_curriculum",   "No\nCurriculum"),
        ("A4_ilp_only",        "ILP-only"),
        ("A4_greedy_only",     "Greedy-only"),
    ]

    gaps = [results[k]["mean_gap_vs_greedy"] for k, _ in variants]
    wins = [results[k]["win_rate"] for k, _ in variants]
    labels = [lbl for _, lbl in variants]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(variants))
    colors = ["#2980b9" if g > 0 else "#e74c3c" for g in gaps]
    bars = ax.bar(x, gaps, 0.6, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)

    ax.axhline(y=0, color="#2c3e50", lw=1.2, ls="--", alpha=0.5)
    ax.set_ylabel("Gap vs. Greedy (%)", fontsize=16)
    ax.set_xlabel("Variant", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title("Ablation Study ($N{=}40$)",
                 fontsize=16, fontweight="bold")

    # Annotate bars with gap% and win rate, positioned outside bars
    for i, (g, w) in enumerate(zip(gaps, wins)):
        sign = "+" if g > 0 else ""
        color = "#2980b9" if g > 0 else "#c0392b"
        va = "bottom" if g >= 0 else "top"
        offset = 0.3 if g >= 0 else -0.3
        ax.text(i, g + offset, f"{sign}{g:.1f}%\n({w:.0f}%)",
                ha="center", va=va, fontsize=13, fontweight="bold", color=color)

    # Pad y-axis so annotations don't clip
    ymin = min(gaps) - 3.5
    ymax = max(gaps) + 3.5
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    savefig(fig, out / "fig_ablations.png")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Fig 1: Graph topology ---
    print("[1/4] Graph topology diagram...")
    fig_graph_topology(OUT_DIR)

    # --- Fig 4: Ablation study ---
    print("\n[2/4] Ablation study...")
    fig_ablations(OUT_DIR)

    # --- Evaluate across available problem sizes (Fig 2 & 3) ---
    configs = [
        (10, MODEL_DIR / "gnn_10obj_best.pt", DATA_DIR / "dataset_10_objects.json"),
        (40, MODEL_DIR / "gnn_final_40obj.pt", DATA_DIR / "dataset_40_objects.json"),
        (200, MODEL_DIR / "gnn_final_200obj.pt", DATA_DIR / "dataset_200_objects.json"),
    ]

    results_by_n = {}
    for n, model_path, dataset_path in configs:
        if not model_path.exists() or not dataset_path.exists():
            print(f"  SKIP N={n}: model or dataset not found")
            continue
        max_sc = 100
        print(f"\n  Evaluating N={n} ({max_sc} val scenarios)...")
        t0 = time.time()
        results_by_n[n] = evaluate_model_on_dataset(model_path, dataset_path, device, max_sc)
        r = results_by_n[n]
        gap = (r["greedy"].mean() - r["gnn"].mean()) / r["greedy"].mean() * 100
        win = (r["gnn"] < r["greedy"]).sum()
        print(f"    GNN={r['gnn'].mean():.1f}  Greedy={r['greedy'].mean():.1f}  "
              f"Gap={gap:.2f}%  Win={win}/{r['n_scenarios']}  ({time.time()-t0:.1f}s)")

    # --- Fig 2: Cost comparison ---
    print("\n[3/4] Cost comparison across sizes...")
    fig_cost_comparison(results_by_n, OUT_DIR)

    # --- Fig 3: Box plot ---
    print("\n[4/4] Box plot of improvements...")
    fig_boxplot(results_by_n, OUT_DIR)

    print(f"\nAll paper figures saved to: {OUT_DIR.relative_to(PROJECT_ROOT)}/")
    print("  fig1_graph_topology.{png,pdf}")
    print("  fig_ablations.{png,pdf}")
    print("  fig2_cost_comparison.{png,pdf}")
    print("  fig2_gap_by_size.{png,pdf}")
    print("  fig3_boxplot_improvements.{png,pdf}")
    print("  fig3_boxplot_gap_pct.{png,pdf}")


if __name__ == "__main__":
    main()
