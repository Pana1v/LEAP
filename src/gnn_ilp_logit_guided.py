"""
[LEGACY — not used by the v8 paper.]
Early LEAP precursor using CBC + MTZ subtour-elimination. Superseded by
`gnn_ilp_circuit.py` (CP-SAT `AddCircuit()` + arc pruning). Kept for
historical reference; do not source new results from it.

Logit-Guided ILP Warm-Start

Uses GNN logits (confidence scores) to:
1. Identify high-confidence decisions (where GNN is very sure)
2. Lock those decisions as hard constraints in ILP
3. Dramatically reduce search space

Expected speedup: 5-20x on medium problems (40-100 objects)

Run with: cd src && python3 gnn_ilp_logit_guided.py --dataset ../data/dataset_40_objects.json --confidence-gap 2.0
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ortools.linear_solver import pywraplp

from gnn_train import (
    GNNPolicy,
    _build_step_graph,
    prepare_scenario as _prepare_scenario,
    load_scenarios as _load_scenarios,
    FEATURE_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_DROPOUT,
)
from gnn_gui import load_model


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════

def load_scenarios(path):
    return _load_scenarios(Path(path))


def prepare_scenario(s, device):
    return _prepare_scenario(s, device)


def select_model_by_count(n: int) -> Path:
    MODEL_DIR = Path(__file__).parent.parent / "models"
    if n <= 10:
        return MODEL_DIR / "gnn_10obj_best.pt"
    elif n <= 40:
        return MODEL_DIR / "gnn_final_40obj.pt"
    else:
        return MODEL_DIR / "gnn_final_200obj.pt"


# ═══════════════════════════════════════════════════════
# GNN ROLLOUT WITH LOGITS
# ═══════════════════════════════════════════════════════

def gnn_rollout_with_logits(
    model, scenario, device
) -> Tuple[float, List[int], List[List[float]]]:
    """
    Run GNN rollout and capture logits at each step.

    Returns:
        (total_cost, sequence, logits_per_step)
        where logits_per_step[i] is logits for step i (length = num_objects)
    """
    model.eval()
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=device)
    robot = scenario["start"].clone()
    cost = 0.0
    route = []
    logits_all = []

    with torch.no_grad():
        for step in range(n):
            nf, ei, om = _build_step_graph(objects, bins, types, mask, robot, device)
            logits = model(nf, ei, om)  # Shape: [n]

            # Store logits for this step
            logits_all.append(logits.cpu().numpy().tolist())

            action = int(torch.argmax(logits).item())
            if not mask[action]:
                break
            sc = (
                torch.norm(robot - objects[action]).item()
                + torch.norm(objects[action] - bins[types[action]]).item()
            )
            cost += sc
            route.append(action)
            robot = bins[types[action]]
            mask[action] = False

    return cost, route, logits_all


def analyze_confidence(logits: List[float], threshold: float = 2.0) -> Tuple[int, float]:
    """
    Analyze GNN confidence at a step.

    Returns:
        (best_action, logit_gap)
        where logit_gap = max_logit - second_max_logit

    High gap (>2.0) means GNN is very confident about this decision.
    """
    logits_np = np.array(logits)
    top2_idx = np.argsort(logits_np)[-2:][::-1]
    gap = logits_np[top2_idx[0]] - logits_np[top2_idx[1]]
    return int(top2_idx[0]), float(gap)


# ═══════════════════════════════════════════════════════
# ILP WITH LOCKED CONSTRAINTS
# ═══════════════════════════════════════════════════════

def solve_ilp_baseline(scenario: Dict, max_objects: int = 60) -> Tuple[float, float]:
    """ILP from scratch (no hints)."""
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    if n == 0:
        return 0.0, 0.0
    if n > max_objects:
        raise ValueError(f"n={n} > max_objects={max_objects}")

    solve_start = time.time()

    pick_bin_cost = float(np.sum(np.linalg.norm(objects - bins[types], axis=1)))
    node_count = n + 1
    start_idx = 0

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    costs = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for j in range(1, node_count):
        obj_idx = j - 1
        costs[start_idx][j] = dist(start, objects[obj_idx])
        costs[j][start_idx] = 0.0

    for i in range(1, node_count):
        obj_i = i - 1
        for j in range(1, node_count):
            if i == j:
                continue
            obj_j = j - 1
            costs[i][j] = dist(bins[types[obj_i]], objects[obj_j])

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("Failed to create CBC solver")

    x = {}
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")

    for i in range(node_count):
        solver.Add(sum(x[i, j] for j in range(node_count) if j != i) == 1)
    for j in range(node_count):
        solver.Add(sum(x[i, j] for i in range(node_count) if i != j) == 1)

    u = [solver.NumVar(0, n, f"u_{i}") for i in range(node_count)]
    solver.Add(u[start_idx] == 0)
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i == j:
                continue
            solver.Add(u[i] - u[j] + 1 <= n * (1 - x[i, j]))

    objective = solver.Sum(
        costs[i][j] * x[i, j]
        for i in range(node_count)
        for j in range(node_count)
        if i != j
    )
    solver.Minimize(objective)

    status = solver.Solve()
    elapsed = time.time() - solve_start

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Baseline failed. Status={status}")

    total_cost = solver.Objective().Value() + pick_bin_cost
    return total_cost, elapsed


def solve_ilp_with_locked_arcs(
    scenario: Dict,
    gnn_sequence: List[int],
    logits_per_step: List[List[float]],
    confidence_gap_threshold: float = 2.0,
    max_objects: int = 60,
) -> Tuple[float, float, int]:
    """
    ILP with high-confidence GNN arcs locked as hard constraints.

    Strategy:
    - For each step, if GNN's logit gap > threshold, lock that decision
    - This dramatically reduces search space

    Returns:
        (cost, elapsed_time, num_locked_arcs)
    """
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    if n == 0:
        return 0.0, 0.0, 0
    if n > max_objects:
        raise ValueError(f"n={n} > max_objects={max_objects}")

    solve_start = time.time()

    pick_bin_cost = float(np.sum(np.linalg.norm(objects - bins[types], axis=1)))
    node_count = n + 1
    start_idx = 0

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    costs = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for j in range(1, node_count):
        obj_idx = j - 1
        costs[start_idx][j] = dist(start, objects[obj_idx])
        costs[j][start_idx] = 0.0

    for i in range(1, node_count):
        obj_i = i - 1
        for j in range(1, node_count):
            if i == j:
                continue
            obj_j = j - 1
            costs[i][j] = dist(bins[types[obj_i]], objects[obj_j])

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("Failed to create CBC solver")

    x = {}
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")

    for i in range(node_count):
        solver.Add(sum(x[i, j] for j in range(node_count) if j != i) == 1)
    for j in range(node_count):
        solver.Add(sum(x[i, j] for i in range(node_count) if i != j) == 1)

    u = [solver.NumVar(0, n, f"u_{i}") for i in range(node_count)]
    solver.Add(u[start_idx] == 0)
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i == j:
                continue
            solver.Add(u[i] - u[j] + 1 <= n * (1 - x[i, j]))

    objective = solver.Sum(
        costs[i][j] * x[i, j]
        for i in range(node_count)
        for j in range(node_count)
        if i != j
    )
    solver.Minimize(objective)

    # ─────────────────────────────────────────────────────
    # Lock high-confidence GNN decisions
    # ─────────────────────────────────────────────────────
    locked_arcs = []
    gnn_nodes = [start_idx]

    for step, obj_idx in enumerate(gnn_sequence):
        if step >= len(logits_per_step):
            break

        logits = logits_per_step[step]
        best_action, gap = analyze_confidence(logits, threshold=confidence_gap_threshold)

        # Only lock if gap is high (GNN is very confident)
        if gap >= confidence_gap_threshold and best_action == obj_idx:
            # This is the arc from current position to next object
            from_node = gnn_nodes[-1]
            to_node = obj_idx + 1  # Object indices are 1-indexed in ILP

            # Lock this arc
            solver.Add(x[from_node, to_node] == 1)
            locked_arcs.append((from_node, to_node))
            gnn_nodes.append(to_node)

    status = solver.Solve()
    elapsed = time.time() - solve_start

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Locked ILP failed. Status={status}")

    total_cost = solver.Objective().Value() + pick_bin_cost
    return total_cost, elapsed, len(locked_arcs)


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Logit-guided ILP with locked constraints"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-scenarios", type=int, default=5)
    parser.add_argument("--confidence-gap", type=float, default=2.0, help="Logit gap threshold for locking")
    parser.add_argument("--max-ilp-objects", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    scenarios = load_scenarios(args.dataset)
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]
    print(f"Loaded {len(scenarios)} scenarios")

    n_objects = max(len(s["objects"]) for s in scenarios)
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = select_model_by_count(n_objects)

    print(f"Loading GNN model: {model_path}")
    device = torch.device(args.device)
    model = load_model(device, str(model_path))
    print(f"Model loaded on {device}\n")

    print("=" * 140)
    print(
        f"{'Idx':<4} {'N':<4} {'GNN Gap':<10} {'Baseline':<12} {'Locked':<12} "
        f"{'Speedup':<12} {'Locked Arcs':<12} {'Confidence':<12}"
    )
    print("=" * 140)

    baseline_times = []
    locked_times = []
    speedups_all = []
    locked_counts = []

    for idx, scenario in enumerate(scenarios):
        scenario_torch = prepare_scenario(scenario, device)

        # GNN rollout with logits
        gnn_cost, gnn_sequence, logits_per_step = gnn_rollout_with_logits(
            model, scenario_torch, device
        )

        # Baseline
        cost_base, time_base = solve_ilp_baseline(scenario, max_objects=args.max_ilp_objects)

        # Locked constraints
        cost_locked, time_locked, num_locked = solve_ilp_with_locked_arcs(
            scenario,
            gnn_sequence,
            logits_per_step,
            confidence_gap_threshold=args.confidence_gap,
            max_objects=args.max_ilp_objects,
        )

        # Verify same optimal cost
        assert abs(cost_base - cost_locked) < 1e-6, f"Cost mismatch: {cost_base} vs {cost_locked}"

        baseline_times.append(time_base)
        locked_times.append(time_locked)

        speedup = time_base / time_locked if time_locked > 0 else float('inf')
        speedups_all.append(speedup)
        locked_counts.append(num_locked)

        gnn_gap_pct = (gnn_cost - cost_base) / cost_base * 100

        print(
            f"{idx:<4} {len(scenario['objects']):<4} {gnn_gap_pct:<10.2f}% "
            f"{time_base:<12.4f} {time_locked:<12.4f} {speedup:<12.2f}x "
            f"{num_locked:<12} {args.confidence_gap:<12.1f}"
        )

    # Summary
    print("\n" + "=" * 140)
    print("SPEEDUP ANALYSIS")
    print("=" * 140)

    baseline_np = np.array(baseline_times)
    locked_np = np.array(locked_times)
    speedups_np = np.array(speedups_all)
    locked_np_arr = np.array(locked_counts)

    print(f"\nBaseline (no warm-start):")
    print(f"  Mean time:  {np.mean(baseline_np):.4f}s")
    print(f"  Total time: {np.sum(baseline_np):.2f}s")

    print(f"\nWith Logit-Guided Locked Constraints:")
    print(f"  Mean time:  {np.mean(locked_np):.4f}s")
    print(f"  Mean speedup: {np.mean(speedups_np):.2f}x")
    print(f"  Speedup range: {np.min(speedups_np):.2f}x to {np.max(speedups_np):.2f}x")
    print(f"  % scenarios improved: {sum(s >= 1.0 for s in speedups_np)/len(speedups_np)*100:.1f}%")

    print(f"\nConstraint Locking:")
    print(f"  Mean arcs locked: {np.mean(locked_np_arr):.1f}")
    print(f"  Max arcs locked:  {np.max(locked_np_arr)}")
    print(f"  Min arcs locked:  {np.min(locked_np_arr)}")

    time_saved = np.sum(baseline_np) - np.sum(locked_np)
    time_saved_pct = (np.sum(baseline_np) - np.sum(locked_np)) / np.sum(baseline_np) * 100

    print(f"\nTotal Time Saved:")
    print(f"  Absolute: {time_saved:.2f}s")
    print(f"  Percent:  {time_saved_pct:.1f}%")

    print("\n" + "=" * 140)
    print("INTERPRETATION")
    print("=" * 140)

    if np.mean(speedups_np) > 1.5:
        print(f"✓ Significant speedup ({np.mean(speedups_np):.1f}x)! Logit-guided locking is effective.")
    elif np.mean(speedups_np) > 1.0:
        print(f"+ Modest speedup ({np.mean(speedups_np):.1f}x). Locking helps but not dramatically.")
    else:
        print(f"✗ No speedup ({np.mean(speedups_np):.1f}x). GNN confidence may not align with optimality.")

    print("\n" + "=" * 140)


if __name__ == "__main__":
    main()
