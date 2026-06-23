"""
Export individual route plots for different methods (Greedy, GNN, ILP)
into subfolders under docs/figures/route_examples/.

Run from src/:  python3 export_route_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gnn_train import (
    GNNPolicy, FEATURE_DIM, DEFAULT_DROPOUT, WORKSPACE_SIZE, NUM_BINS,
    _build_step_graph,
)

PROJECT_ROOT = Path(__file__).parent.parent
OUT_ROOTS = [
    PROJECT_ROOT / "docs" / "figures" / "route_examples",
    PROJECT_ROOT / "references" / "figures" / "route_examples",
]
OUT_ROOT = OUT_ROOTS[0]  # primary; we mirror into all entries below

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
BIN_LABELS = ["Bin 0", "Bin 1", "Bin 2", "Bin 3"]
GNN_COLOR = "#2980b9"
NN_COLOR = "#e74c3c"
NC_COLOR = "#e67e22"
ILP_COLOR = "#8e44ad"

DATASETS = {
    5: {
        "data": PROJECT_ROOT / "data" / "dataset_5_objects.json",
        "model": PROJECT_ROOT / "models" / "gnn_ilp.pt",
    },
    10: {
        "data": PROJECT_ROOT / "data" / "dataset_10_objects.json",
        "model": PROJECT_ROOT / "models" / "gnn_10obj_best.pt",
    },
    20: {
        "data": PROJECT_ROOT / "data" / "dataset_20_objects.json",
        "model": PROJECT_ROOT / "models" / "gnn_ilp.pt",
    },
    40: {
        "data": PROJECT_ROOT / "data" / "dataset_40_objects.json",
        "model": PROJECT_ROOT / "models" / "gnn_final_40obj.pt",
    },
}

SCENARIO_INDICES = [0, 42, 99]

VAL_SPLIT = 0.1
SEED = 0


def _val_indices(n_total: int) -> list:
    """Reproduce gnn_train.split_dataset's val partition deterministically."""
    rng = np.random.default_rng(SEED)
    perm = np.arange(n_total)
    rng.shuffle(perm)
    val_size = int(n_total * VAL_SPLIT)
    return sorted(int(i) for i in perm[:val_size])


def _select_val_scenarios(raw_data, model, device):
    """Pick three representative val scenarios for route plots:
       (median GNN improvement, best GNN improvement, ILP-prefix-available best).

       Returns a list of (raw_index, label) tuples.
    """
    val_idx = _val_indices(len(raw_data))
    n = len(raw_data[0]["objects"])
    prefix_key = str(n)
    records = []
    for ri in val_idx:
        scen = raw_data[ri]
        s = prepare_scenario(scen, device)
        nc_cost, _ = greedy_nc_rollout(s)
        gnn_cost, _ = gnn_rollout(model, s, device)
        improvement = (nc_cost - gnn_cost) / nc_cost * 100
        has_ilp = prefix_key in scen.get("ilp_prefixes", {})
        records.append({
            "raw_index": ri,
            "improvement": improvement,
            "nc_cost": nc_cost,
            "gnn_cost": gnn_cost,
            "has_ilp": has_ilp,
        })
    records.sort(key=lambda r: r["improvement"])
    median = records[len(records) // 2]
    best = records[-1]
    best_with_ilp = next(
        (r for r in reversed(records) if r["has_ilp"]),
        best,
    )
    selected = []
    seen = set()
    for r, label in [(median, "median"), (best, "best"), (best_with_ilp, "best_with_ilp")]:
        if r["raw_index"] in seen:
            continue
        seen.add(r["raw_index"])
        selected.append((r["raw_index"], label, r["improvement"]))
    return selected


def load_model(path, device):
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


def prepare_scenario(scenario, device):
    return {
        "objects": torch.tensor(scenario["objects"], dtype=torch.float32, device=device),
        "types": torch.tensor(scenario["types"], dtype=torch.long, device=device),
        "bins": torch.tensor(scenario["bins"], dtype=torch.float32, device=device),
        "start": torch.tensor(scenario["start"], dtype=torch.float32, device=device),
    }


def gnn_rollout(model, s, device):
    objects, bins, types = s["objects"], s["bins"], s["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=device)
    robot = s["start"].clone()
    cost, route = 0.0, []
    with torch.no_grad():
        for _ in range(n):
            nf, ei, om = _build_step_graph(objects, bins, types, mask, robot, device)
            logits = model(nf, ei, om)
            action = int(torch.argmax(logits).item())
            if not mask[action]:
                break
            cost += (torch.norm(robot - objects[action]).item()
                     + torch.norm(objects[action] - bins[types[action]]).item())
            route.append(action)
            robot = bins[types[action]]
            mask[action] = False
    return cost, route


def greedy_nn_rollout(s):
    """Nearest Neighbor: pick the closest object (pick-leg only)."""
    objects, bins, types = s["objects"], s["bins"], s["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=objects.device)
    robot = s["start"].clone()
    cost, route = 0.0, []
    for _ in range(n):
        valid = torch.nonzero(mask, as_tuple=False).flatten()
        if valid.numel() == 0:
            break
        pick_d = torch.norm(objects[valid] - robot.unsqueeze(0), dim=1)
        best = valid[torch.argmin(pick_d)]
        cost += (torch.norm(robot - objects[best]).item()
                 + torch.norm(objects[best] - bins[types[best]]).item())
        route.append(best.item())
        robot = bins[types[best]]
        mask[best] = False
    return cost, route


def greedy_nc_rollout(s):
    """Nearest Cycle: pick the object minimizing pick + place cost."""
    objects, bins, types = s["objects"], s["bins"], s["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=objects.device)
    robot = s["start"].clone()
    cost, route = 0.0, []
    for _ in range(n):
        valid = torch.nonzero(mask, as_tuple=False).flatten()
        if valid.numel() == 0:
            break
        pick_d = torch.norm(objects[valid] - robot.unsqueeze(0), dim=1)
        place_d = torch.norm(objects[valid] - bins[types[valid]], dim=1)
        best = valid[torch.argmin(pick_d + place_d)]
        cost += (torch.norm(robot - objects[best]).item()
                 + torch.norm(objects[best] - bins[types[best]]).item())
        route.append(best.item())
        robot = bins[types[best]]
        mask[best] = False
    return cost, route


def ilp_route_cost(scenario_raw, s):
    n = len(scenario_raw["objects"])
    prefix_key = str(n)
    prefixes = scenario_raw.get("ilp_prefixes", {})
    if prefix_key not in prefixes:
        return None, None
    order = prefixes[prefix_key]
    objects, bins, types = s["objects"], s["bins"], s["types"]
    robot = s["start"].clone()
    cost = 0.0
    for idx in order:
        cost += (torch.norm(robot - objects[idx]).item()
                 + torch.norm(objects[idx] - bins[types[idx]]).item())
        robot = bins[types[idx]]
    return cost, order


def draw_route(scenario_tensor, route, title, cost, method_color, out_path,
               step_numbers=True):
    objects = scenario_tensor["objects"].cpu().numpy()
    bins_arr = scenario_tensor["bins"].cpu().numpy()
    types_np = scenario_tensor["types"].cpu().numpy()
    start = scenario_tensor["start"].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, WORKSPACE_SIZE + 5)
    ax.set_ylim(-5, WORKSPACE_SIZE + 5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f9fa")

    for v in [0, 25, 50, 75, 100]:
        ax.axhline(y=v, color="#ecf0f1", lw=0.5)
        ax.axvline(x=v, color="#ecf0f1", lw=0.5)

    for i, (bx, by) in enumerate(bins_arr):
        ax.plot(bx, by, "s", color=BIN_COLORS[i], ms=16, zorder=5,
                markeredgecolor="white", mew=2)
        ax.annotate(BIN_LABELS[i], (bx, by), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=12, fontweight="bold",
                    color=BIN_COLORS[i])

    for i, (ox, oy) in enumerate(objects):
        ax.plot(ox, oy, "o", color=BIN_COLORS[types_np[i]], ms=7, alpha=0.7,
                zorder=3, markeredgecolor="white", mew=0.4)

    pos = start.copy()
    for step, obj_idx in enumerate(route):
        obj = objects[obj_idx]
        t = types_np[obj_idx]
        bin_pos = bins_arr[t]
        ax.annotate("", xy=obj, xytext=pos,
                    arrowprops=dict(arrowstyle="-|>", color=method_color,
                                    lw=1.0, alpha=0.5))
        ax.annotate("", xy=bin_pos, xytext=obj,
                    arrowprops=dict(arrowstyle="-|>", color=BIN_COLORS[t],
                                    lw=0.8, alpha=0.4, linestyle="dashed"))
        if step_numbers:
            ax.annotate(str(step + 1), (obj[0], obj[1]),
                        textcoords="offset points", xytext=(-7, -11),
                        fontsize=10, fontweight="bold", color=method_color,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=method_color, alpha=0.85),
                        zorder=10)
        pos = bin_pos

    ax.plot(*start, "*", color="#f1c40f", ms=18, zorder=6,
            markeredgecolor="#2c3e50", mew=1.2)
    ax.annotate("START", start, textcoords="offset points", xytext=(10, -10),
                fontsize=12, fontweight="bold", color="#2c3e50")

    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.tick_params(labelsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  -> {out_path.relative_to(PROJECT_ROOT)}")
    # Mirror into the additional roots (e.g. references/figures/) so LaTeX
    # picks up the regenerated PNG without a manual copy.
    rel = out_path.relative_to(OUT_ROOTS[0])
    for extra_root in OUT_ROOTS[1:]:
        mirrored = extra_root / rel
        mirrored.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(mirrored, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  -> {mirrored.relative_to(PROJECT_ROOT)}")
    plt.close(fig)


def main():
    device = torch.device("cpu")

    for n_obj, cfg in DATASETS.items():
        if not cfg["data"].exists():
            print(f"Skipping N={n_obj}: dataset not found at {cfg['data']}")
            continue
        if not cfg["model"].exists():
            print(f"Skipping N={n_obj}: model not found at {cfg['model']}")
            continue

        print(f"\n=== N={n_obj} ===")
        with open(cfg["data"]) as f:
            raw_data = json.load(f)

        model = load_model(cfg["model"], device)

        # For N=10 use val-set selection (representative + ILP-available);
        # other sizes keep the existing fixed indices for backward compatibility.
        if n_obj == 10:
            val_selected = _select_val_scenarios(raw_data, model, device)
            scenarios_to_plot = [(ri, lbl) for ri, lbl, _ in val_selected]
            print(f"  Selected val scenarios: " +
                  ", ".join(f"#{ri} ({lbl}, +{imp:.2f}%)"
                            for ri, lbl, imp in val_selected))
        else:
            scenarios_to_plot = [(si, "fixed") for si in SCENARIO_INDICES]

        for si, scenario_label in scenarios_to_plot:
            if si >= len(raw_data):
                continue
            scenario_raw = raw_data[si]
            s = prepare_scenario(scenario_raw, device)
            n = len(scenario_raw["objects"])
            tag = f"n{n_obj}_s{si}"

            nn_cost, nn_route = greedy_nn_rollout(s)
            nc_cost, nc_route = greedy_nc_rollout(s)
            gnn_cost, gnn_route = gnn_rollout(model, s, device)
            ilp_cost, ilp_route = ilp_route_cost(scenario_raw, s)

            base = OUT_ROOT / f"{n_obj}_objects" / f"scenario_{si}"

            draw_route(s, nn_route,
                       f"Nearest Neighbor (NN), {n_obj} objects, scenario {si}\nCost: {nn_cost:.1f}",
                       nn_cost, NN_COLOR,
                       base / "greedy_nn.png")

            draw_route(s, nc_route,
                       f"Nearest Cycle (NC), {n_obj} objects, scenario {si}\nCost: {nc_cost:.1f}",
                       nc_cost, NC_COLOR,
                       base / "greedy_nc.png")

            saving_vs_nc = (nc_cost - gnn_cost) / nc_cost * 100
            draw_route(s, gnn_route,
                       f"GNN, {n_obj} objects, scenario {si}\nCost: {gnn_cost:.1f} ({saving_vs_nc:+.1f}% vs NC)",
                       gnn_cost, GNN_COLOR,
                       base / "gnn.png")

            if ilp_route is not None:
                ilp_saving = (nc_cost - ilp_cost) / nc_cost * 100
                draw_route(s, ilp_route,
                           f"ILP Optimal, {n_obj} objects, scenario {si}\nCost: {ilp_cost:.1f} ({ilp_saving:+.1f}% vs NC)",
                           ilp_cost, ILP_COLOR,
                           base / "ilp.png")

            print(f"  Scenario {si}: NN={nn_cost:.1f}, NC={nc_cost:.1f}, GNN={gnn_cost:.1f}"
                  + (f", ILP={ilp_cost:.1f}" if ilp_cost else ""))

    print(f"\nAll plots saved to {OUT_ROOT.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
