"""
[LEGACY-BASELINE — used by v8 §V-B-2 as the warm-start negative-result baseline.]
Implements the CBC + MTZ + `SetHint()` pathway that v8 reports as
"0.93–1.06× (neutral)". Superseded for the main LEAP results by
`gnn_ilp_circuit.py` (CP-SAT `AddCircuit()` + arc pruning). Do not port
to Circuit — that would erase the §V-B-2 comparison.

GNN-Warm-Started ILP Hybrid Solver (Optimal)

Combines GNN approximate solution with ILP exact solver:
1. Run GNN to get a candidate sequence
2. Use sequence as warm-start hint for ILP solver
3. Solve ILP to optimality (no time limit)
4. Always return optimal solution

The GNN warm-start accelerates ILP solving by providing a good initial solution.

Run with: cd src && python3 gnn_ilp_hybrid.py --dataset ../data/dataset_40_objects.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ortools.linear_solver import pywraplp

from gnn_train import (
    GNNPolicy,
    _build_step_graph,
    load_scenarios as _load_scenarios,
    prepare_scenario as _prepare_scenario,
    FEATURE_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_DROPOUT,
)
from gnn_gui import gnn_rollout_with_route, load_model


# ═══════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════

import time

MODEL_DIR = Path(__file__).parent.parent / "models"


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════

def load_scenarios(path):
    return _load_scenarios(Path(path))


def prepare_scenario(s, device):
    return _prepare_scenario(s, device)


def select_model_by_count(n: int) -> Path:
    """Auto-select GNN model based on object count."""
    if n <= 10:
        return MODEL_DIR / "gnn_10obj_best.pt"
    elif n <= 40:
        return MODEL_DIR / "gnn_final_40obj.pt"
    else:
        return MODEL_DIR / "gnn_final_200obj.pt"


# ═══════════════════════════════════════════════════════
# ILP SOLVER WITH WARM-START
# ═══════════════════════════════════════════════════════

def solve_ilp_route_warm(
    scenario: Dict,
    gnn_sequence: List[int],
    max_objects: int = 60,
) -> Tuple[float, List[int], float]:
    """
    Solve ILP with GNN sequence as warm-start hint.

    Solves to optimality (no time limit). The GNN warm-start accelerates solving.

    Args:
        scenario: Dataset scenario dict with keys: objects, bins, types, start
        gnn_sequence: List of 0-indexed object indices (GNN proposed order)
        max_objects: Skip ILP if n > max_objects

    Returns:
        (cost, order_objects, elapsed_time) where:
            cost: float (optimal cost)
            order_objects: List[int] (optimal object order)
            elapsed_time: float (seconds taken to solve)
    """
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    if n == 0:
        return 0.0, [], 0.0

    if n > max_objects:
        raise ValueError(f"Scenario has {n} objects, exceeds max_objects={max_objects}")

    solve_start_time = time.time()

    # Constant term: pick-to-bin legs are order-independent
    pick_bin_cost = float(np.sum(np.linalg.norm(objects - bins[types], axis=1)))

    # Node 0 is the start position; nodes 1..n correspond to objects 0..n-1
    node_count = n + 1
    start_idx = 0

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    # Cost matrix
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

    # Decision variables: x[i,j] = 1 if arc i->j is in tour
    x = {}
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")

    # Degree constraints: exactly one outgoing and one incoming per node
    for i in range(node_count):
        solver.Add(sum(x[i, j] for j in range(node_count) if j != i) == 1)
    for j in range(node_count):
        solver.Add(sum(x[i, j] for i in range(node_count) if i != j) == 1)

    # MTZ subtour elimination for nodes 1..n
    u = [solver.NumVar(0, n, f"u_{i}") for i in range(node_count)]
    solver.Add(u[start_idx] == 0)
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i == j:
                continue
            solver.Add(u[i] - u[j] + 1 <= n * (1 - x[i, j]))

    # Objective
    objective = solver.Sum(
        costs[i][j] * x[i, j]
        for i in range(node_count)
        for j in range(node_count)
        if i != j
    )
    solver.Minimize(objective)

    # ─────────────────────────────────────────────────────
    # WARM-START: Inject GNN sequence as initial hint
    # This accelerates ILP solving by providing a good starting point
    # ─────────────────────────────────────────────────────
    if gnn_sequence and len(gnn_sequence) > 0:
        # Build arc set from GNN sequence
        # GNN sequence: [obj_a, obj_b, ...] (0-indexed)
        # ILP nodes: 0 = start, i+1 = object i
        gnn_nodes = [start_idx] + [s + 1 for s in gnn_sequence]

        active_arcs = set()
        for i in range(len(gnn_nodes) - 1):
            active_arcs.add((gnn_nodes[i], gnn_nodes[i + 1]))

        hint_vars, hint_vals = [], []
        for (i, j), var in x.items():
            hint_vars.append(var)
            hint_vals.append(1.0 if (i, j) in active_arcs else 0.0)

        solver.SetHint(hint_vars, hint_vals)

    # Solve to optimality (no time limit)
    status = solver.Solve()

    # Check result status — must be OPTIMAL
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(
            f"ILP solver failed to reach OPTIMAL. Status={status}. "
            f"This should not happen unless there's a bug or the problem is infeasible."
        )

    # Recover order by following edges from start
    order_nodes = []
    current = start_idx
    visited = set([start_idx])
    while True:
        next_j = None
        for j in range(node_count):
            if j == current:
                continue
            if x[current, j].solution_value() > 0.5:
                next_j = j
                break
        if next_j is None or next_j == start_idx:
            break
        order_nodes.append(next_j)
        if next_j in visited:
            break
        visited.add(next_j)
        current = next_j

    order_objects = [node - 1 for node in order_nodes]
    total_cost = solver.Objective().Value() + pick_bin_cost
    elapsed = time.time() - solve_start_time
    return total_cost, order_objects, elapsed


# ═══════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════

def compute_sequence_cost(scenario: Dict, sequence: List[int]) -> float:
    """Compute total cost for a given sequence."""
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    if not sequence:
        return 0.0

    robot = start.copy()
    total_cost = 0.0
    for obj_idx in sequence:
        total_cost += np.linalg.norm(robot - objects[obj_idx])
        total_cost += np.linalg.norm(objects[obj_idx] - bins[types[obj_idx]])
        robot = bins[types[obj_idx]].copy()
    return total_cost


def pct_delta(baseline, candidate):
    """Compute percentage improvement: (baseline - candidate) / baseline * 100."""
    baseline = np.asarray(baseline, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (baseline - candidate) / baseline * 100.0
    result = np.where(baseline == 0, 0.0, result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="GNN-Warm-Started ILP Hybrid Solver"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--model", type=str, default=None, help="Path to GNN model (auto-selected if omitted)")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Max scenarios to evaluate")
    parser.add_argument("--max-ilp-objects", type=int, default=60, help="Object cap for ILP solver")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Load data
    print(f"Loading dataset: {args.dataset}")
    scenarios = load_scenarios(args.dataset)
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]
    print(f"Loaded {len(scenarios)} scenarios")

    # Infer object count
    n_objects = max(len(s["objects"]) for s in scenarios)
    print(f"Max objects in dataset: {n_objects}")

    # Select and load GNN model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = select_model_by_count(n_objects)
    print(f"Loading GNN model: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device(args.device)
    model = load_model(device, str(model_path))
    print(f"Model loaded on {device}")

    # ─────────────────────────────────────────────────────
    # Evaluation loop
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(
        f"{'Idx':<4} {'N':<4} {'Greedy':<10} {'GNN':<10} {'Optimal':<10} "
        f"{'GNN Δ%':<8} {'Optimal Δ%':<10} {'ILP Time (s)':<12}"
    )
    print("=" * 110)

    greedy_costs = []
    gnn_costs = []
    optimal_costs = []
    ilp_times = []

    for idx, scenario in enumerate(scenarios):
        # Prepare scenario for GNN
        scenario_torch = prepare_scenario(scenario, device)

        # GNN rollout
        gnn_cost, gnn_sequence = gnn_rollout_with_route(model, scenario_torch, device)

        # ILP warm-started with GNN sequence — solves to optimality
        optimal_cost, optimal_sequence, ilp_time = solve_ilp_route_warm(
            scenario,
            gnn_sequence,
            max_objects=args.max_ilp_objects,
        )

        greedy_cost = scenario["greedy_cost"]
        greedy_costs.append(greedy_cost)
        gnn_costs.append(gnn_cost)
        optimal_costs.append(optimal_cost)
        ilp_times.append(ilp_time)

        gnn_delta = pct_delta(greedy_cost, gnn_cost)
        optimal_delta = pct_delta(greedy_cost, optimal_cost)

        print(
            f"{idx:<4} {len(scenario['objects']):<4} {greedy_cost:<10.2f} "
            f"{gnn_cost:<10.2f} {optimal_cost:<10.2f} {gnn_delta:<8.2f} "
            f"{optimal_delta:<10.2f} {ilp_time:<12.3f}"
        )

    # ─────────────────────────────────────────────────────
    # Summary statistics
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("SUMMARY STATISTICS")
    print("=" * 110)

    greedy_np = np.array(greedy_costs, dtype=np.float32)
    gnn_np = np.array(gnn_costs, dtype=np.float32)
    optimal_np = np.array(optimal_costs, dtype=np.float32)
    times_np = np.array(ilp_times, dtype=np.float32)

    print(f"Scenarios evaluated: {len(scenarios)}")
    print(f"\nMean costs:")
    print(f"  Greedy (baseline):     {np.mean(greedy_np):.3f}")
    print(f"  GNN (approximate):     {np.mean(gnn_np):.3f}")
    print(f"  ILP+GNN warm (optimal):{np.mean(optimal_np):.3f}")

    gnn_vs_greedy = pct_delta(greedy_np, gnn_np)
    optimal_vs_greedy = pct_delta(greedy_np, optimal_np)
    optimal_vs_gnn = pct_delta(gnn_np, optimal_np)

    print(f"\nMean improvement vs Greedy:")
    print(f"  GNN:                  {np.mean(gnn_vs_greedy):+.2f}%")
    print(f"  ILP (warm-started):   {np.mean(optimal_vs_greedy):+.2f}%")

    print(f"\nILP improvement over GNN:")
    print(f"  Mean Δ:               {np.mean(optimal_vs_gnn):+.2f}%")
    print(f"  Min Δ:                {np.min(optimal_vs_gnn):+.2f}%")
    print(f"  Max Δ:                {np.max(optimal_vs_gnn):+.2f}%")

    print(f"\nILP Solver Timing (with GNN warm-start):")
    print(f"  Mean time:            {np.mean(times_np):.3f}s")
    print(f"  Min time:             {np.min(times_np):.3f}s")
    print(f"  Max time:             {np.max(times_np):.3f}s")
    print(f"  Total time:           {np.sum(times_np):.1f}s")

    print("\n" + "=" * 110)


if __name__ == "__main__":
    main()
