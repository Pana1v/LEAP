"""
C4: Evaluate GNN vs cached ILP costs on the full N=10 validation set.

Dataset has ilp_costs[str(10)] for all scenarios, so we can report the
GNN's mean ILP gap on the full val split rather than the 13/50 subset
that the 60s re-solve timeout yielded.

Run from src/:  python3 eval_n10_ilp_gap.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from gnn_train import (
    DEFAULT_DROPOUT,
    FEATURE_DIM,
    GNNPolicy,
    load_scenarios,
    prepare_scenario,
    rollout_model,
    split_dataset,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA = PROJECT_ROOT / "data" / "dataset_10_objects.json"
MODEL = PROJECT_ROOT / "models" / "gnn_10obj_best.pt"
OUT = PROJECT_ROOT / "experiments" / "n10_ilp_gap.json"
SEED = 0
VAL_SPLIT = 0.1


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
    m = GNNPolicy(FEATURE_DIM, hd, heads, dp).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


def main():
    device = torch.device("cpu")
    scenarios = load_scenarios(DATA)
    _, val_raw = split_dataset(scenarios, VAL_SPLIT, SEED)
    print(f"Val scenarios: {len(val_raw)}")

    val = [prepare_scenario(s, device) for s in val_raw]
    model = load_model(MODEL, device)

    gnn_costs, ilp_costs, greedy_costs = [], [], []
    with torch.no_grad():
        for s in val:
            cost = rollout_model(model, s, device)
            gnn_costs.append(float(cost))
            ilp_costs.append(float(s["ilp_costs"]["10"]))
            greedy_costs.append(float(s["greedy_cost"]))

    gnn_c = np.array(gnn_costs)
    ilp_c = np.array(ilp_costs)
    gre_c = np.array(greedy_costs)

    gap_vs_ilp = (gnn_c - ilp_c) / ilp_c * 100.0
    gap_vs_greedy = (gre_c - gnn_c) / gre_c * 100.0
    win = float((gnn_c < gre_c).mean() * 100.0)

    summary = {
        "n_scenarios": int(len(val_raw)),
        "gnn_mean_cost": float(gnn_c.mean()),
        "ilp_mean_cost": float(ilp_c.mean()),
        "greedy_mean_cost": float(gre_c.mean()),
        "mean_gap_vs_ilp_pct": float(gap_vs_ilp.mean()),
        "std_gap_vs_ilp_pct": float(gap_vs_ilp.std()),
        "median_gap_vs_ilp_pct": float(np.median(gap_vs_ilp)),
        "p95_gap_vs_ilp_pct": float(np.percentile(gap_vs_ilp, 95)),
        "mean_gap_vs_greedy_pct": float(gap_vs_greedy.mean()),
        "win_rate_pct": win,
    }
    print(json.dumps(summary, indent=2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
