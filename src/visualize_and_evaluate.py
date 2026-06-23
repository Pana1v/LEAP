"""
Comprehensive GNN evaluation and visualization for the 40-object final model.
Individual plots organized into subfolders.
Run from src/:  python3 visualize_and_evaluate.py
"""

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from gnn_train import (
    GNNPolicy, FEATURE_DIM, DEFAULT_DROPOUT, WORKSPACE_SIZE,
    load_scenarios, split_dataset, prepare_scenario,
    rollout_model, evaluate, _build_step_graph,
)

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "gnn_dataset_40_objects_20260407_210044"
OUT_ROOT = PROJECT_ROOT / "docs" / "figures" / "40obj"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "gnn_final_40obj.pt"
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_40_objects.json"

EVAL_SIZE = 100
BIN_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
BIN_NAMES = ["Bin 0 (TL)", "Bin 1 (TR)", "Bin 2 (BL)", "Bin 3 (BR)"]

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
})


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.relative_to(PROJECT_ROOT)}")


def load_training_log(log_dir: Path):
    rows = []
    with open(log_dir / "training_log.csv") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if v else 0.0 for k, v in row.items()})
    return rows


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


def gnn_rollout_with_route(model, scenario, device):
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=device)
    robot = scenario["start"].clone()
    cost = 0.0
    route, step_costs = [], []
    with torch.no_grad():
        for _ in range(n):
            nf, ei, om = _build_step_graph(objects, bins, types, mask, robot, device)
            logits = model(nf, ei, om)
            action = int(torch.argmax(logits).item())
            if not mask[action]:
                break
            sc = (torch.norm(robot - objects[action]).item() +
                  torch.norm(objects[action] - bins[types[action]]).item())
            cost += sc
            step_costs.append(sc)
            route.append(action)
            robot = bins[types[action]]
            mask[action] = False
    return cost, route, step_costs


def greedy_rollout_with_route(scenario):
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=objects.device)
    robot = scenario["start"].clone()
    cost = 0.0
    route, step_costs = [], []
    for _ in range(n):
        valid = torch.nonzero(mask, as_tuple=False).flatten()
        if valid.numel() == 0:
            break
        pick_d = torch.norm(objects[valid] - robot.unsqueeze(0), dim=1)
        place_d = torch.norm(objects[valid] - bins[types[valid]], dim=1)
        best = valid[torch.argmin(pick_d + place_d)]
        sc = (torch.norm(robot - objects[best]).item() +
              torch.norm(objects[best] - bins[types[best]]).item())
        cost += sc
        step_costs.append(sc)
        route.append(best.item())
        robot = bins[types[best]]
        mask[best] = False
    return cost, route, step_costs


# helpers for stage shading
def _get_stage_meta(rows):
    stages = [int(r["stage"]) for r in rows]
    stage_colors = {5: "#e74c3c", 10: "#3498db", 20: "#2ecc71"}
    boundaries = []
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            boundaries.append(i + 0.5)
    return stages, stage_colors, boundaries


def _shade(ax, rows):
    epochs = [r["epoch"] for r in rows]
    stages, colors, boundaries = _get_stage_meta(rows)
    prev = 0.5
    for boundary in boundaries + [len(epochs) + 0.5]:
        stage = stages[min(int(prev), len(stages) - 1)]
        ax.axvspan(prev, boundary, alpha=0.08, color=colors.get(stage, "gray"))
        if boundary <= len(epochs):
            ax.axvline(x=boundary, color="gray", ls="--", alpha=0.3, lw=0.8)
        prev = boundary


# ═══════════════════════════════════════════════════════
# 1. TRAINING PLOTS  (out_root/training/)
# ═══════════════════════════════════════════════════════

def plot_training(rows, out: Path):
    out = out / "training"
    epochs = [r["epoch"] for r in rows]

    # --- loss ---
    fig, ax = plt.subplots(figsize=(8, 5))
    losses = [r["train_loss"] for r in rows]
    ax.plot(epochs, losses, "o-", color="#2c3e50", lw=2.5, ms=6)
    ax.fill_between(epochs, losses, alpha=0.1, color="#2c3e50")
    _shade(ax, rows)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Loss", fontsize=16, fontweight="bold", pad=10)
    pct = (losses[0] - losses[-1]) / losses[0] * 100
    ax.annotate(f"{pct:.0f}% reduction", xy=(epochs[-1], losses[-1]),
                xytext=(-60, 30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#2c3e50"),
                fontsize=10, fontweight="bold", color="#2c3e50")
    savefig(fig, out / "loss.png")

    # --- val cost comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    val_costs = [r["val_mean_cost"] for r in rows]
    greedy_cost = val_costs[0] / (1 - rows[0]["val_mean_gap_vs_greedy"] / 100)
    ax.plot(epochs, val_costs, "o-", color="#2980b9", lw=2.5, ms=6, label="GNN")
    ax.axhline(y=greedy_cost, color="#e74c3c", ls=":", lw=2, label=f"Greedy ({greedy_cost:.0f})")
    _shade(ax, rows)
    best_idx = int(np.argmin(val_costs))
    ax.annotate(f"Best: {val_costs[best_idx]:.0f}", xy=(epochs[best_idx], val_costs[best_idx]),
                xytext=(15, -25), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#2980b9"),
                fontsize=10, fontweight="bold", color="#2980b9")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Route Cost")
    ax.set_title("Validation Cost Comparison")
    ax.legend()
    savefig(fig, out / "val_cost.png")

    # --- gap vs greedy ---
    fig, ax = plt.subplots(figsize=(8, 5))
    gaps = [r["val_mean_gap_vs_greedy"] for r in rows]
    colors = ["#27ae60" if g > 0 else "#e74c3c" for g in gaps]
    ax.bar(epochs, gaps, color=colors, alpha=0.8, edgecolor="white", width=0.7)
    ax.axhline(y=0, color="#2c3e50", lw=1.5)
    _shade(ax, rows)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Improvement over Greedy (%)")
    ax.set_title("Gap vs Greedy per Epoch")
    ax.annotate(f"Final: {gaps[-1]:.1f}%", xy=(epochs[-1], gaps[-1]),
                xytext=(-50, 15), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#27ae60")
    savefig(fig, out / "gap_vs_greedy.png")

    # --- win rate ---
    fig, ax = plt.subplots(figsize=(8, 5))
    wins = [r["val_win_rate"] for r in rows]
    bar_c = ["#27ae60" if w >= 90 else "#f39c12" if w >= 50 else "#e74c3c" for w in wins]
    ax.bar(epochs, wins, color=bar_c, alpha=0.85, edgecolor="white", width=0.7)
    ax.set_ylim(0, 110)
    _shade(ax, rows)
    for i, (e, w) in enumerate(zip(epochs, wins)):
        if i == 0 or i == len(epochs) - 1:
            ax.text(e, w + 2, f"{w:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate vs Greedy", fontsize=16, fontweight="bold", pad=10)
    savefig(fig, out / "win_rate.png")

    # --- learning rate ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["learning_rate"] for r in rows], "o-", color="#8e44ad", lw=2, ms=5)
    _shade(ax, rows)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    savefig(fig, out / "learning_rate.png")

    # --- gradient norm ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["grad_norm"] for r in rows], "o-", color="#d35400", lw=2, ms=5)
    _shade(ax, rows)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    savefig(fig, out / "grad_norm.png")


# ═══════════════════════════════════════════════════════
# 2. EVALUATION PLOTS  (out_root/evaluation/)
# ═══════════════════════════════════════════════════════

def plot_evaluation(gnn_costs, greedy_costs, gaps, out: Path):
    out = out / "evaluation"
    n = len(gnn_costs)
    savings = greedy_costs - gnn_costs

    # --- scatter ---
    fig, ax = plt.subplots(figsize=(8, 7))
    wins = gnn_costs < greedy_costs
    ax.scatter(greedy_costs[wins], gnn_costs[wins], alpha=0.4, s=15, c="#27ae60",
               edgecolors="none", label=f"GNN wins ({wins.sum()})")
    ax.scatter(greedy_costs[~wins], gnn_costs[~wins], alpha=0.6, s=25, c="#e74c3c",
               marker="x", label=f"Greedy wins ({(~wins).sum()})")
    lims = [min(gnn_costs.min(), greedy_costs.min()) * 0.95,
            max(gnn_costs.max(), greedy_costs.max()) * 1.05]
    ax.plot(lims, lims, "--", color="#2c3e50", lw=1.5, alpha=0.5, label="Break-even")
    ax.set_xlabel("Greedy Cost"); ax.set_ylabel("GNN Cost")
    ax.set_title("Per-Scenario: GNN vs Greedy")
    ax.legend()
    savefig(fig, out / "scatter_gnn_vs_greedy.png")

    # --- histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins_hist = np.linspace(gaps.min() - 0.5, gaps.max() + 0.5, 35)
    _, _, patches = ax.hist(gaps, bins=bins_hist, alpha=0.8, edgecolor="white")
    for patch, left in zip(patches, bins_hist[:-1]):
        patch.set_facecolor("#27ae60" if left >= 0 else "#e74c3c")
    ax.axvline(x=0, color="#2c3e50", ls="--", lw=2, alpha=0.7)
    ax.axvline(x=np.mean(gaps), color="#8e44ad", ls="-", lw=2.5,
               label=f"Mean: {np.mean(gaps):.2f}%")
    ax.axvline(x=np.median(gaps), color="#2980b9", ls="-.", lw=2,
               label=f"Median: {np.median(gaps):.2f}%")
    ax.set_xlabel("Improvement over Greedy (%)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Improvements")
    ax.legend()
    savefig(fig, out / "histogram_improvement.png")

    # --- CDF ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_gaps = np.sort(gaps)
    cdf = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps) * 100
    ax.plot(sorted_gaps, cdf, color="#2980b9", lw=2.5)
    ax.fill_betweenx(cdf, sorted_gaps, 0, where=sorted_gaps > 0, alpha=0.1, color="#27ae60")
    ax.axvline(x=0, color="#e74c3c", ls="--", lw=1.5, alpha=0.5)
    win_pct = (gaps > 0).sum() / n * 100
    ax.axhline(y=100 - win_pct, color="#95a5a6", ls=":", lw=1)
    ax.annotate(f"{win_pct:.1f}% beat greedy",
                xy=(0, 100 - win_pct), xytext=(2, 100 - win_pct - 15),
                fontsize=10, fontweight="bold", color="#27ae60",
                arrowprops=dict(arrowstyle="->", color="#27ae60"))
    ax.set_xlabel("Improvement over Greedy (%)"); ax.set_ylabel("Cumulative %")
    ax.set_title("Cumulative Distribution of Improvement")
    savefig(fig, out / "cdf_improvement.png")

    # --- absolute savings waterfall ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_idx = np.argsort(savings)[::-1]
    ax.bar(np.arange(n), savings[sorted_idx], width=1.0,
           color=["#27ae60" if s > 0 else "#e74c3c" for s in savings[sorted_idx]],
           alpha=0.7, edgecolor="none")
    ax.axhline(y=0, color="#2c3e50", lw=1)
    ax.set_xlabel("Scenario (sorted by savings)"); ax.set_ylabel("Cost Saved (Greedy - GNN)")
    ax.set_title("Absolute Cost Savings per Scenario")
    ax.annotate(f"Total saved: {savings.sum():,.0f}\nMean: {savings.mean():.1f}/scenario",
                xy=(0.02, 0.95), xycoords="axes fraction", fontsize=10,
                fontweight="bold", color="#27ae60", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", alpha=0.8))
    savefig(fig, out / "savings_waterfall.png")

    # --- difficulty bins ---
    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = 8
    greedy_sorted_idx = np.argsort(greedy_costs)
    bin_size = n // n_bins
    labels, gm, grem = [], [], []
    for i in range(n_bins):
        sl = greedy_sorted_idx[i * bin_size:(i + 1) * bin_size]
        gc = greedy_costs[sl].mean()
        labels.append(f"{gc:.0f}")
        grem.append(gc)
        gm.append(gnn_costs[sl].mean())
    x_pos = np.arange(n_bins)
    w = 0.35
    ax.bar(x_pos - w/2, grem, w, color="#e74c3c", alpha=0.7, label="Greedy")
    ax.bar(x_pos + w/2, gm, w, color="#2980b9", alpha=0.7, label="GNN")
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=45, fontsize=10)
    ax.set_xlabel("Scenario Difficulty (Greedy Cost)"); ax.set_ylabel("Mean Cost")
    ax.set_title("Performance by Scenario Difficulty")
    ax.legend()
    savefig(fig, out / "difficulty_breakdown.png")


# ═══════════════════════════════════════════════════════
# 3. ROUTE VISUALIZATIONS  (out_root/routes/)
# ═══════════════════════════════════════════════════════

def _draw_route(ax, scenario, route, title, cost, color_main="#2980b9"):
    objects = scenario["objects"].cpu().numpy()
    bins_arr = scenario["bins"].cpu().numpy()
    types_np = scenario["types"].cpu().numpy()
    start = scenario["start"].cpu().numpy()

    ax.set_xlim(-5, WORKSPACE_SIZE + 5)
    ax.set_ylim(-5, WORKSPACE_SIZE + 5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f9fa")

    for v in [0, 25, 50, 75, 100]:
        ax.axhline(y=v, color="#ecf0f1", lw=0.5)
        ax.axvline(x=v, color="#ecf0f1", lw=0.5)

    for i, (bx, by) in enumerate(bins_arr):
        ax.plot(bx, by, "s", color=BIN_COLORS[i], ms=14, zorder=5,
                markeredgecolor="white", mew=1.5)
        ax.annotate(f"B{i}", (bx, by), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, fontweight="bold", color=BIN_COLORS[i])

    for i, (ox, oy) in enumerate(objects):
        ax.plot(ox, oy, "o", color=BIN_COLORS[types_np[i]], ms=5, alpha=0.6,
                zorder=3, markeredgecolor="white", mew=0.3)

    pos = start.copy()
    for obj_idx in route:
        obj = objects[obj_idx]
        t = types_np[obj_idx]
        bin_pos = bins_arr[t]
        ax.annotate("", xy=obj, xytext=pos,
                     arrowprops=dict(arrowstyle="-|>", color=color_main, lw=0.6, alpha=0.4))
        ax.annotate("", xy=bin_pos, xytext=obj,
                     arrowprops=dict(arrowstyle="-|>", color=BIN_COLORS[t], lw=0.6, alpha=0.4))
        pos = bin_pos

    ax.plot(*start, "*", color="#f1c40f", ms=15, zorder=6, markeredgecolor="#2c3e50", mew=1)
    ax.set_title(f"{title}\nCost: {cost:.1f}", fontsize=11, fontweight="bold")


def plot_routes(scenarios, model, device, gnn_costs, greedy_costs, gaps, out: Path):
    out = out / "routes"
    sorted_idx = np.argsort(gaps)
    cases = [
        ("best_win", sorted_idx[-1], "Best GNN Win"),
        ("median", sorted_idx[len(sorted_idx) // 2], "Median Case"),
        ("worst", sorted_idx[0], "Worst Case"),
    ]

    for tag, idx, label in cases:
        s = scenarios[idx]
        gnn_cost, gnn_route, _ = gnn_rollout_with_route(model, s, device)
        greedy_cost, greedy_route, _ = greedy_rollout_with_route(s)
        saving = (greedy_cost - gnn_cost) / greedy_cost * 100

        # GNN route
        fig, ax = plt.subplots(figsize=(8, 8))
        _draw_route(ax, s, gnn_route, f"GNN: {label} (saves {saving:.1f}%)", gnn_cost, "#2980b9")
        savefig(fig, out / f"{tag}_gnn.png")

        # Greedy route
        fig, ax = plt.subplots(figsize=(8, 8))
        _draw_route(ax, s, greedy_route, f"Greedy: {label}", greedy_cost, "#e74c3c")
        savefig(fig, out / f"{tag}_greedy.png")


# ═══════════════════════════════════════════════════════
# 4. STEP-LEVEL ANALYSIS  (out_root/steps/)
# ═══════════════════════════════════════════════════════

def plot_steps(scenarios, model, device, out: Path):
    out = out / "steps"
    n_sample = min(50, len(scenarios))
    all_gnn, all_greedy = [], []
    for s in scenarios[:n_sample]:
        _, _, gsc = gnn_rollout_with_route(model, s, device)
        _, _, grsc = greedy_rollout_with_route(s)
        all_gnn.append(gsc)
        all_greedy.append(grsc)

    max_steps = max(max(len(x) for x in all_gnn), max(len(x) for x in all_greedy))

    def pad_stats(all_steps):
        padded = np.full((len(all_steps), max_steps), np.nan)
        for i, sc in enumerate(all_steps):
            padded[i, :len(sc)] = sc
        return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)

    gnn_mean, gnn_std = pad_stats(all_gnn)
    greedy_mean, greedy_std = pad_stats(all_greedy)
    steps = np.arange(1, max_steps + 1)

    # --- cost per step ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, gnn_mean, color="#2980b9", lw=2, label="GNN")
    ax.fill_between(steps, gnn_mean - gnn_std, gnn_mean + gnn_std, alpha=0.15, color="#2980b9")
    ax.plot(steps, greedy_mean, color="#e74c3c", lw=2, label="Greedy")
    ax.fill_between(steps, greedy_mean - greedy_std, greedy_mean + greedy_std, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Pick Step"); ax.set_ylabel("Step Cost")
    ax.set_title("Average Cost per Step")
    ax.legend()
    savefig(fig, out / "cost_per_step.png")

    # --- cumulative ---
    fig, ax = plt.subplots(figsize=(10, 5))
    gnn_cum = np.nancumsum(gnn_mean)
    greedy_cum = np.nancumsum(greedy_mean)
    ax.plot(steps, gnn_cum, color="#2980b9", lw=2.5, label="GNN")
    ax.plot(steps, greedy_cum, color="#e74c3c", lw=2.5, label="Greedy")
    ax.fill_between(steps, gnn_cum, greedy_cum, alpha=0.15, color="#27ae60", label="GNN savings")
    ax.set_xlabel("Pick Step"); ax.set_ylabel("Cumulative Cost")
    ax.set_title("Cumulative Cost over Steps")
    ax.legend()
    savefig(fig, out / "cumulative_cost.png")

    # --- per-step advantage ---
    fig, ax = plt.subplots(figsize=(10, 5))
    advantage = greedy_mean - gnn_mean
    ax.bar(steps, advantage, width=0.8,
           color=["#27ae60" if a > 0 else "#e74c3c" for a in advantage],
           alpha=0.7, edgecolor="none")
    ax.axhline(y=0, color="#2c3e50", lw=1)
    ax.set_xlabel("Pick Step"); ax.set_ylabel("Greedy Cost - GNN Cost")
    ax.set_title("Per-Step Advantage (positive = GNN better)", fontsize=20,
                 fontweight="bold", pad=10)
    early = advantage[:10].mean()
    late = advantage[-10:].mean()
    ax.annotate(f"Early avg: {early:+.1f}", xy=(5, early), fontsize=10,
                fontweight="bold", color="#27ae60" if early > 0 else "#e74c3c")
    ax.annotate(f"Late avg: {late:+.1f}", xy=(max_steps - 8, late), fontsize=10,
                fontweight="bold", color="#27ae60" if late > 0 else "#e74c3c")
    savefig(fig, out / "per_step_advantage.png")


# ═══════════════════════════════════════════════════════
# 5. TYPE / BIN ANALYSIS  (out_root/types/)
# ═══════════════════════════════════════════════════════

def plot_types(scenarios, gnn_costs, greedy_costs, gaps, out: Path):
    out = out / "types"
    type_counts = np.array([np.bincount(s["types"].cpu().numpy(), minlength=4) for s in scenarios])

    # --- type distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(4):
        ax.hist(type_counts[:, i], bins=15, alpha=0.5, color=BIN_COLORS[i],
                label=BIN_NAMES[i], edgecolor="white")
    ax.set_xlabel("Number of Objects of Type"); ax.set_ylabel("Number of Scenarios")
    ax.set_title("Type Distribution across Scenarios")
    ax.legend(fontsize=10)
    savefig(fig, out / "type_distribution.png")

    # --- improvement vs imbalance ---
    fig, ax = plt.subplots(figsize=(8, 5))
    imbalance = type_counts.std(axis=1)
    ax.scatter(imbalance, gaps, alpha=0.3, s=15, c="#2980b9", edgecolors="none")
    z = np.polyfit(imbalance, gaps, 1)
    x_t = np.linspace(imbalance.min(), imbalance.max(), 50)
    ax.plot(x_t, np.poly1d(z)(x_t), "--", color="#e74c3c", lw=2, label=f"Trend (slope={z[0]:.2f})")
    ax.set_xlabel("Type Count Std (imbalance)"); ax.set_ylabel("Improvement over Greedy (%)")
    ax.set_title("Effect of Type Imbalance on GNN Advantage")
    ax.legend()
    savefig(fig, out / "imbalance_vs_improvement.png")

    # --- boxplot by dominant type ---
    fig, ax = plt.subplots(figsize=(8, 5))
    dominant = type_counts.argmax(axis=1)
    for t in range(4):
        mask = dominant == t
        if mask.sum() > 0:
            data = gaps[mask]
            bp = ax.boxplot([data], positions=[t], widths=0.5,
                            patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(BIN_COLORS[t])
            bp["boxes"][0].set_alpha(0.6)
            bp["medians"][0].set_color("#2c3e50")
            bp["medians"][0].set_linewidth(2)
            ax.text(t, np.median(data) + 0.15, f"n={mask.sum()}", ha="center",
                    fontsize=10, color="#7f8c8d")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"Type {i}" for i in range(4)])
    ax.set_xlabel("Dominant Object Type"); ax.set_ylabel("Improvement over Greedy (%)")
    ax.set_title("Performance by Dominant Type")
    savefig(fig, out / "boxplot_by_dominant_type.png")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- 1. Training curves ---
    print("[1/5] Training plots...")
    rows = load_training_log(LOG_DIR)
    plot_training(rows, OUT_ROOT)

    # --- 2. Load & evaluate ---
    print(f"\n[2/5] Evaluating {EVAL_SIZE} val scenarios...")
    scenarios_raw = load_scenarios(DATASET_PATH)
    _, val_raw = split_dataset(scenarios_raw, val_split=0.1, seed=0)
    val_raw = val_raw[:EVAL_SIZE]
    scenarios = [prepare_scenario(s, device) for s in val_raw]
    model = load_model(MODEL_PATH, device)

    gnn_costs, greedy_costs = [], []
    t0 = time.time()
    with torch.no_grad():
        for s in scenarios:
            gnn_costs.append(rollout_model(model, s, device))
            greedy_costs.append(s["greedy_cost"])
    elapsed = time.time() - t0
    gnn_costs = np.array(gnn_costs)
    greedy_costs = np.array(greedy_costs)
    gaps = (greedy_costs - gnn_costs) / greedy_costs * 100
    print(f"  {len(scenarios)} scenarios in {elapsed:.1f}s | gap={gaps.mean():.2f}% | wins={(gaps>0).sum()}/{len(gaps)}")

    # --- 3-5. Plots ---
    print("\n[3/5] Evaluation plots...")
    plot_evaluation(gnn_costs, greedy_costs, gaps, OUT_ROOT)

    print("\n[4/5] Route comparisons...")
    plot_routes(scenarios, model, device, gnn_costs, greedy_costs, gaps, OUT_ROOT)

    print("\n[5/5] Step + type analysis...")
    plot_steps(scenarios, model, device, OUT_ROOT)
    plot_types(scenarios, gnn_costs, greedy_costs, gaps, OUT_ROOT)

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"  RESULTS — 40 Objects ({len(scenarios)} val scenarios)")
    print(f"{'='*55}")
    print(f"  GNN Mean Cost:      {gnn_costs.mean():.1f}")
    print(f"  Greedy Mean Cost:   {greedy_costs.mean():.1f}")
    print(f"  Mean Improvement:   {gaps.mean():.2f}%")
    print(f"  Win Rate:           {(gaps>0).sum()}/{len(gaps)} ({(gaps>0).sum()/len(gaps)*100:.1f}%)")
    print(f"  Best / Worst:       {gaps.max():+.2f}% / {gaps.min():+.2f}%")
    print(f"\n  Output: {OUT_ROOT.relative_to(PROJECT_ROOT)}/")
    print(f"    training/   (6 plots)")
    print(f"    evaluation/ (5 plots)")
    print(f"    routes/     (6 plots)")
    print(f"    steps/      (3 plots)")
    print(f"    types/      (3 plots)")


if __name__ == "__main__":
    main()
