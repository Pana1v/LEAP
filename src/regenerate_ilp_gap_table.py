"""
Regenerate the per-sample ILP-gap table (paper Table II / `tab:ilp_gap`).

The original Table II claimed per-sample ILP solves at N=40 in 76--538 ms,
which is impossible: CBC at N=40 routinely exceeds 60 s per instance and
typically returns no solution within practical timeouts. We therefore
re-target Table II to N=10, where the dataset already stores exact ILP
optimal costs for every scenario (no solver call needed at evaluation
time). The GNN rollout uses `models/gnn_10obj_best.pt`.

Run from `src/`:
    python3 regenerate_ilp_gap_table.py [--n 10]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from gnn_train import (
    GNNPolicy,
    FEATURE_DIM,
    DEFAULT_DROPOUT,
    _build_step_graph,
    load_scenarios,
    split_dataset,
    prepare_scenario,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_10_objects.json"
MODEL_PATH = PROJECT_ROOT / "models" / "gnn_10obj_best.pt"
OUT_PATH = PROJECT_ROOT / "experiments" / "ilp_gap_per_sample_10obj.json"
SEED = 0
VAL_SPLIT = 0.1
ILP_PREFIX_KEY = "10"  # full N=10 ILP optimal cost is stored in dataset


def load_model(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
        hd = int(blob.get("hidden_dim", 128))
        heads = int(blob.get("heads", 4))
        dp = float(blob.get("dropout", DEFAULT_DROPOUT))
    else:
        sd = blob
        hd = 128
        heads = 4
        dp = DEFAULT_DROPOUT
    model = GNNPolicy(FEATURE_DIM, hd, heads, dp).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def gnn_rollout_timed(model, scenario, device):
    """Run GNN rollout and return (cost, elapsed_ms)."""
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    mask = torch.ones(objects.size(0), device=device, dtype=torch.bool)
    robot = scenario["start"].clone()
    cost = 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(objects.size(0)):
            nf, ei, om = _build_step_graph(objects, bins, types, mask, robot, device)
            logits = model(nf, ei, om)
            action = int(torch.argmax(logits).item())
            if not mask[action]:
                break
            cost += (
                torch.norm(robot - objects[action]).item()
                + torch.norm(objects[action] - bins[types[action]]).item()
            )
            robot = bins[types[action]]
            mask[action] = False
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return cost, elapsed_ms


def ilp_solve_timed(scenario_raw, timeout_s):
    """Solve N=40 ILP with CBC. Returns (cost, elapsed_ms, status_str)."""
    try:
        from ortools.linear_solver import pywraplp
    except ImportError:
        return None, None, "ortools_unavailable"

    objects = scenario_raw["objects"]
    types = scenario_raw["types"]
    bins_pos = scenario_raw["bins"]
    start = scenario_raw["start"]
    n = len(objects)

    t0 = time.perf_counter()
    solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        return None, None, "solver_init_failed"
    solver.SetTimeLimit(int(timeout_s * 1000))

    obj_pos = [(o[0], o[1]) for o in objects]
    bin_pos_list = [(bins_pos[types[j]][0], bins_pos[types[j]][1]) for j in range(n)]

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    x = {(i, j): solver.IntVar(0, 1, f"x_{i}_{j}") for i in range(n) for j in range(n)}

    for i in range(n):
        solver.Add(sum(x[i, j] for j in range(n)) == 1)
    for j in range(n):
        solver.Add(sum(x[i, j] for i in range(n)) == 1)

    total_cost = solver.Sum([])
    for j in range(n):
        pick = dist((start[0], start[1]), obj_pos[j])
        place = dist(obj_pos[j], bin_pos_list[j])
        total_cost += x[0, j] * (pick + place)

    for i in range(1, n):
        for j in range(n):
            place_j = dist(obj_pos[j], bin_pos_list[j])
            for k in range(n):
                if k == j:
                    continue
                pick = dist(bin_pos_list[k], obj_pos[j])
                y = solver.IntVar(0, 1, f"y_{i}_{k}_{j}")
                solver.Add(y <= x[i - 1, k])
                solver.Add(y <= x[i, j])
                solver.Add(y >= x[i - 1, k] + x[i, j] - 1)
                total_cost += y * (pick + place_j)

    solver.Minimize(total_cost)
    status = solver.Solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None, elapsed_ms, "no_solution"

    seq = []
    for i in range(n):
        for j in range(n):
            if x[i, j].solution_value() > 0.5:
                seq.append(j)
                break
    cost = solver.Objective().Value()
    status_str = "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible"
    return cost, elapsed_ms, status_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="Number of scenarios to display")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu | cuda | auto",
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Model:  {MODEL_PATH}")
    print(f"Data:   {DATA_PATH}")

    scenarios_raw = load_scenarios(DATA_PATH)
    train_raw, val_raw = split_dataset(scenarios_raw, VAL_SPLIT, SEED)
    print(f"Train: {len(train_raw)}  Val: {len(val_raw)}")

    # Keep only val scenarios that have an exact full-N ILP cost stored
    val_with_ilp = [
        s for s in val_raw
        if ILP_PREFIX_KEY in s.get("ilp_costs", {})
        and s["ilp_costs"][ILP_PREFIX_KEY] is not None
    ]
    print(f"Val with stored ILP-{ILP_PREFIX_KEY} cost: {len(val_with_ilp)}")

    val_with_ilp.sort(key=lambda s: s.get("greedy_cost", 1e9))
    # Select evenly spaced scenarios across the difficulty range (by greedy cost)
    n_avail = len(val_with_ilp)
    indices = np.linspace(0, n_avail - 1, args.n, dtype=int)
    selected_raw = [val_with_ilp[i] for i in indices]
    print(f"Selected {len(selected_raw)} val scenarios (evenly spaced by greedy cost)")

    model = load_model(MODEL_PATH, device)

    # Warmup
    with torch.no_grad():
        warm = prepare_scenario(selected_raw[0], device)
        for _ in range(5):
            _ = gnn_rollout_timed(model, warm, device)

    rows = []
    for idx, raw in enumerate(selected_raw, start=1):
        s = prepare_scenario(raw, device)
        gnn_cost, gnn_ms = gnn_rollout_timed(model, s, device)
        ilp_cost = float(raw["ilp_costs"][ILP_PREFIX_KEY])
        gap = (gnn_cost - ilp_cost) / ilp_cost * 100
        rows.append(
            {
                "sample": idx,
                "greedy_cost_stored": raw.get("greedy_cost"),
                "gnn_cost": gnn_cost,
                "gnn_time_ms": gnn_ms,
                "ilp_cost": ilp_cost,
                "gap_pct": gap,
            }
        )
        print(
            f"[{idx:2d}] GNN={gnn_cost:.1f} ({gnn_ms:.2f}ms)  "
            f"ILP={ilp_cost:.1f}  gap={gap:+.2f}%  "
            f"greedy_stored={raw.get('greedy_cost'):.1f}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "device": str(device),
                "model": str(MODEL_PATH),
                "data": str(DATA_PATH),
                "ilp_source": "stored ilp_costs in dataset (full-N optimum)",
                "selection_strategy": f"{args.n} evenly-spaced-by-greedy-cost val scenarios with full ILP",
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {OUT_PATH}")

    print("\n=== LaTeX rows for tab:ilp_gap ===")
    for r in rows:
        print(
            f"{r['sample']:<2} & {r['gnn_time_ms']:.2f} & "
            f"{r['gnn_cost']:.2f} & {r['ilp_cost']:.2f} & {r['gap_pct']:.2f}\\% \\\\"
        )

    gaps = [r["gap_pct"] for r in rows]
    print(
        f"\nSummary over {len(rows)} samples: "
        f"min={min(gaps):.2f}%  max={max(gaps):.2f}%  mean={sum(gaps)/len(gaps):.2f}%"
    )


if __name__ == "__main__":
    main()
