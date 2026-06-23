"""
[LEGACY — not used by the v8 paper.]
Early CP-SAT prototype using the MTZ subtour-elimination formulation.
Superseded by `gnn_ilp_circuit.py`, which uses the native `AddCircuit()`
global constraint. Kept for historical reference only; do not source new
results from it.

CP-SAT vs CBC: Branch & Bound Comparison

Benchmarks four approaches to see which actually accelerates B&B:
1. CBC cold-start (no hints)
2. CBC + SetHint()
3. CP-SAT cold-start (no hints)
4. CP-SAT + AddHint()

Key insight: CP-SAT's AddHint() is used in LNS (Large Neighborhood Search) workers.
CBC's SetHint() just provides an incumbent bound. CP-SAT should be 2-10x faster with hints.

Run with: cd src && python3 gnn_ilp_cpsat.py --dataset ../data/dataset_40_objects.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from gnn_train import prepare_scenario as _prepare_scenario
from gnn_gui import gnn_rollout_with_route, load_model


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════

def load_scenarios(path):
    from gnn_train import load_scenarios as _load_scenarios
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
# SHARED: Build cost matrix
# ═══════════════════════════════════════════════════════

def build_cost_matrix(scenario: Dict, max_objects: int = 60):
    """Build cost matrix and metadata for TSP."""
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    if n == 0:
        return None, None, None, None
    if n > max_objects:
        raise ValueError(f"n={n} > max_objects={max_objects}")

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

    return costs, pick_bin_cost, node_count, start_idx


# ═══════════════════════════════════════════════════════
# CBC SOLVERS
# ═══════════════════════════════════════════════════════

def solve_cbc_cold(scenario: Dict, max_objects: int = 60) -> Tuple[float, float]:
    """CBC cold-start (no hints)."""
    costs, pick_bin_cost, node_count, start_idx = build_cost_matrix(scenario, max_objects)
    if costs is None:
        return 0.0, 0.0

    t0 = time.time()

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

    u = [solver.NumVar(0, node_count - 1, f"u_{i}") for i in range(node_count)]
    solver.Add(u[start_idx] == 0)
    n = node_count - 1
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i == j:
                continue
            solver.Add(u[i] - u[j] + 1 <= n * (1 - x[i, j]))

    objective = solver.Sum(
        costs[i][j] * x[i, j] for i in range(node_count) for j in range(node_count) if i != j
    )
    solver.Minimize(objective)
    status = solver.Solve()
    elapsed = time.time() - t0

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"CBC cold failed. Status={status}")

    total_cost = solver.Objective().Value() + pick_bin_cost
    return total_cost, elapsed


def solve_cbc_hint(scenario: Dict, gnn_sequence: List[int], max_objects: int = 60) -> Tuple[float, float]:
    """CBC + SetHint()."""
    costs, pick_bin_cost, node_count, start_idx = build_cost_matrix(scenario, max_objects)
    if costs is None:
        return 0.0, 0.0

    t0 = time.time()

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

    u = [solver.NumVar(0, node_count - 1, f"u_{i}") for i in range(node_count)]
    solver.Add(u[start_idx] == 0)
    n = node_count - 1
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i == j:
                continue
            solver.Add(u[i] - u[j] + 1 <= n * (1 - x[i, j]))

    objective = solver.Sum(
        costs[i][j] * x[i, j] for i in range(node_count) for j in range(node_count) if i != j
    )
    solver.Minimize(objective)

    # SetHint
    if gnn_sequence:
        gnn_nodes = [start_idx] + [s + 1 for s in gnn_sequence]
        active_arcs = set()
        for i in range(len(gnn_nodes) - 1):
            active_arcs.add((gnn_nodes[i], gnn_nodes[i + 1]))

        hint_vars, hint_vals = [], []
        for (i, j), var in x.items():
            hint_vars.append(var)
            hint_vals.append(1.0 if (i, j) in active_arcs else 0.0)
        solver.SetHint(hint_vars, hint_vals)

    status = solver.Solve()
    elapsed = time.time() - t0

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"CBC hint failed. Status={status}")

    total_cost = solver.Objective().Value() + pick_bin_cost
    return total_cost, elapsed


# ═══════════════════════════════════════════════════════
# CP-SAT SOLVERS
# ═══════════════════════════════════════════════════════

COST_SCALE = 10000  # Scale float costs to integers (10x for better precision)


def solve_cpsat_cold(scenario: Dict, max_objects: int = 60) -> Tuple[float, float]:
    """CP-SAT cold-start (no hints)."""
    costs, pick_bin_cost, node_count, start_idx = build_cost_matrix(scenario, max_objects)
    if costs is None:
        return 0.0, 0.0

    t0 = time.time()

    model = cp_model.CpModel()

    # Scale costs to integers
    int_costs = [[int(round(costs[i][j] * COST_SCALE)) for j in range(node_count)] for i in range(node_count)]

    # Binary arc variables
    x = {}
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

    # Degree constraints
    for i in range(node_count):
        model.AddExactlyOne(x[i, j] for j in range(node_count) if j != i)
    for j in range(node_count):
        model.AddExactlyOne(x[i, j] for i in range(node_count) if i != j)

    # MTZ subtour elimination
    u = [model.NewIntVar(0, node_count - 1, f"u_{i}") for i in range(node_count)]
    model.Add(u[start_idx] == 0)
    n = node_count - 1
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i == j:
                continue
            model.Add(u[i] - u[j] + 1 <= n * (1 - x[i, j]))

    # Objective (integer scaled)
    model.Minimize(sum(int_costs[i][j] * x[i, j] for i in range(node_count) for j in range(node_count) if i != j))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    status = solver.Solve(model)
    elapsed = time.time() - t0

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT cold failed. Status={status}")

    scaled_obj = solver.ObjectiveValue()
    total_cost = (scaled_obj / COST_SCALE) + pick_bin_cost
    return total_cost, elapsed


def solve_cpsat_hint(scenario: Dict, gnn_sequence: List[int], max_objects: int = 60) -> Tuple[float, float]:
    """CP-SAT + AddHint() (hints used in LNS workers)."""
    costs, pick_bin_cost, node_count, start_idx = build_cost_matrix(scenario, max_objects)
    if costs is None:
        return 0.0, 0.0

    t0 = time.time()

    model = cp_model.CpModel()

    # Scale costs to integers
    int_costs = [[int(round(costs[i][j] * COST_SCALE)) for j in range(node_count)] for i in range(node_count)]

    # Binary arc variables
    x = {}
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

    # Degree constraints
    for i in range(node_count):
        model.AddExactlyOne(x[i, j] for j in range(node_count) if j != i)
    for j in range(node_count):
        model.AddExactlyOne(x[i, j] for i in range(node_count) if i != j)

    # MTZ subtour elimination
    u = [model.NewIntVar(0, node_count - 1, f"u_{i}") for i in range(node_count)]
    model.Add(u[start_idx] == 0)
    n = node_count - 1
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i == j:
                continue
            model.Add(u[i] - u[j] + 1 <= n * (1 - x[i, j]))

    # Objective (integer scaled)
    model.Minimize(sum(int_costs[i][j] * x[i, j] for i in range(node_count) for j in range(node_count) if i != j))

    # AddHint - used in LNS workers in CP-SAT
    if gnn_sequence:
        gnn_nodes = [start_idx] + [s + 1 for s in gnn_sequence]
        active_arcs = set()
        for i in range(len(gnn_nodes) - 1):
            active_arcs.add((gnn_nodes[i], gnn_nodes[i + 1]))

        for (i, j), var in x.items():
            hint_value = 1 if (i, j) in active_arcs else 0
            model.AddHint(var, hint_value)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    status = solver.Solve(model)
    elapsed = time.time() - t0

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT hint failed. Status={status}")

    scaled_obj = solver.ObjectiveValue()
    total_cost = (scaled_obj / COST_SCALE) + pick_bin_cost
    return total_cost, elapsed


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CBC vs CP-SAT: Branch & Bound Comparison")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-scenarios", type=int, default=5)
    parser.add_argument("--max-ilp-objects", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    scenarios = load_scenarios(args.dataset)
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]
    print(f"Loaded {len(scenarios)} scenarios\n")

    n_objects = max(len(s["objects"]) for s in scenarios)
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = select_model_by_count(n_objects)

    print(f"Loading GNN model: {model_path}")
    device = torch.device(args.device)
    model = load_model(device, str(model_path))
    print(f"Model loaded on {device}\n")

    print("=" * 160)
    print(
        f"{'Idx':<4} {'N':<4} {'CBC Cold':<12} {'CBC Hint':<12} {'CP-SAT Cold':<12} "
        f"{'CP-SAT Hint':<12} {'Hint Speedup':<15} {'SAT Speedup':<15}"
    )
    print("=" * 160)

    cbc_cold_times = []
    cbc_hint_times = []
    cpsat_cold_times = []
    cpsat_hint_times = []

    for idx, scenario in enumerate(scenarios):
        scenario_torch = prepare_scenario(scenario, device)
        gnn_cost, gnn_sequence = gnn_rollout_with_route(model, scenario_torch, device)

        # Four solvers
        cost_cbc_cold, time_cbc_cold = solve_cbc_cold(scenario, max_objects=args.max_ilp_objects)
        cost_cbc_hint, time_cbc_hint = solve_cbc_hint(scenario, gnn_sequence, max_objects=args.max_ilp_objects)
        cost_cpsat_cold, time_cpsat_cold = solve_cpsat_cold(scenario, max_objects=args.max_ilp_objects)
        cost_cpsat_hint, time_cpsat_hint = solve_cpsat_hint(scenario, gnn_sequence, max_objects=args.max_ilp_objects)

        # Verify same cost (allow small tolerance due to integer scaling)
        assert abs(cost_cbc_cold - cost_cbc_hint) < 1e-2, f"CBC costs differ: {cost_cbc_cold} vs {cost_cbc_hint}"
        assert abs(cost_cbc_cold - cost_cpsat_cold) < 1.0, f"CBC vs CP-SAT cold differ: {cost_cbc_cold} vs {cost_cpsat_cold}"
        assert abs(cost_cbc_cold - cost_cpsat_hint) < 1.0, f"CBC vs CP-SAT hint differ: {cost_cbc_cold} vs {cost_cpsat_hint}"

        cbc_cold_times.append(time_cbc_cold)
        cbc_hint_times.append(time_cbc_hint)
        cpsat_cold_times.append(time_cpsat_cold)
        cpsat_hint_times.append(time_cpsat_hint)

        hint_speedup_cbc = time_cbc_cold / time_cbc_hint if time_cbc_hint > 0 else float('inf')
        sat_speedup = time_cpsat_cold / time_cpsat_hint if time_cpsat_hint > 0 else float('inf')

        print(
            f"{idx:<4} {len(scenario['objects']):<4} {time_cbc_cold:<12.4f} "
            f"{time_cbc_hint:<12.4f} {time_cpsat_cold:<12.4f} {time_cpsat_hint:<12.4f} "
            f"{hint_speedup_cbc:<15.2f}x {sat_speedup:<15.2f}x"
        )

    # Summary
    print("\n" + "=" * 160)
    print("SUMMARY: WHICH APPROACH IS FASTEST?")
    print("=" * 160)

    cbc_cold_np = np.array(cbc_cold_times)
    cbc_hint_np = np.array(cbc_hint_times)
    cpsat_cold_np = np.array(cpsat_cold_times)
    cpsat_hint_np = np.array(cpsat_hint_times)

    print(f"\nCBC Cold-Start:")
    print(f"  Mean: {np.mean(cbc_cold_np):.4f}s (baseline)")

    print(f"\nCBC + SetHint():")
    print(f"  Mean: {np.mean(cbc_hint_np):.4f}s")
    print(f"  Speedup vs CBC cold: {np.mean(cbc_cold_np) / np.mean(cbc_hint_np):.2f}x")

    print(f"\nCP-SAT Cold-Start:")
    print(f"  Mean: {np.mean(cpsat_cold_np):.4f}s")
    print(f"  vs CBC cold: {np.mean(cbc_cold_np) / np.mean(cpsat_cold_np):.2f}x")

    print(f"\nCP-SAT + AddHint():")
    print(f"  Mean: {np.mean(cpsat_hint_np):.4f}s")
    print(f"  Speedup vs CP-SAT cold: {np.mean(cpsat_cold_np) / np.mean(cpsat_hint_np):.2f}x")
    print(f"  vs CBC cold: {np.mean(cbc_cold_np) / np.mean(cpsat_hint_np):.2f}x")

    fastest = min(
        ("CBC cold", np.mean(cbc_cold_np)),
        ("CBC hint", np.mean(cbc_hint_np)),
        ("CP-SAT cold", np.mean(cpsat_cold_np)),
        ("CP-SAT hint", np.mean(cpsat_hint_np)),
    )

    print(f"\n🏆 FASTEST: {fastest[0]} ({fastest[1]:.4f}s)")

    if "CP-SAT hint" in fastest[0]:
        print("\n✓ CP-SAT + AddHint() wins! Hints actually work in CP-SAT's LNS workers.")
    elif "CP-SAT" in fastest[0]:
        print("\n+ CP-SAT faster, but hints don't help much (may need tuning).")
    else:
        print("\n✗ CBC faster. CP-SAT overhead may not be worth it for this problem size.")

    print("\n" + "=" * 160)


if __name__ == "__main__":
    main()
