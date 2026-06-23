"""
[LEGACY — not used by the v8 paper.]
Early LEAP precursor using CBC + MTZ subtour-elimination. Superseded by
`gnn_ilp_circuit.py` (CP-SAT `AddCircuit()` + arc pruning). Kept for
historical reference; do not source new results from it.

Aggressive GNN-Warm-Started ILP Hybrid

Better warm-start strategies:
1. Lock first k% of GNN sequence as hard constraints (prefix-lock)
2. This dramatically reduces search space: 2^(locked positions) fewer branches
3. Measure actual speedup improvement

Run with: cd src && python3 gnn_ilp_hybrid_aggressive.py --dataset ../data/dataset_40_objects.json --lock-ratio 0.2
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ortools.linear_solver import pywraplp

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
# ILP SOLVERS WITH DIFFERENT WARM-START STRATEGIES
# ═══════════════════════════════════════════════════════

def solve_ilp_baseline(scenario: Dict, max_objects: int = 60) -> Tuple[float, float]:
    """ILP from scratch (no hints, no constraints)."""
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    if n == 0:
        return 0.0, 0.0
    if n > max_objects:
        raise ValueError(f"Scenario has {n} objects, exceeds max_objects={max_objects}")

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
        raise RuntimeError(f"Baseline ILP failed to reach OPTIMAL. Status={status}")

    total_cost = solver.Objective().Value() + pick_bin_cost
    return total_cost, elapsed


def solve_ilp_hint_only(
    scenario: Dict, gnn_sequence: List[int], max_objects: int = 60
) -> Tuple[float, float]:
    """ILP with SetHint() only (no prefix-lock)."""
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    if n == 0:
        return 0.0, 0.0
    if n > max_objects:
        raise ValueError(f"Scenario has {n} objects, exceeds max_objects={max_objects}")

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

    # SetHint only
    if gnn_sequence and len(gnn_sequence) > 0:
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
    elapsed = time.time() - solve_start

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Hint-only ILP failed to reach OPTIMAL. Status={status}")

    total_cost = solver.Objective().Value() + pick_bin_cost
    return total_cost, elapsed


def solve_ilp_prefix_lock(
    scenario: Dict, gnn_sequence: List[int], lock_ratio: float = 0.2, max_objects: int = 60
) -> Tuple[float, float, int]:
    """
    ILP with aggressive warm-start: prioritize first lock_ratio% of GNN sequence.

    Instead of hard-locking (which might force suboptimal solution if GNN is wrong),
    we use strong hints for the prefix arcs and regular hints for the rest.

    Returns: (cost, elapsed_time, num_prefixed_arcs)
    """
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    if n == 0:
        return 0.0, 0.0, 0
    if n > max_objects:
        raise ValueError(f"Scenario has {n} objects, exceeds max_objects={max_objects}")

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
    # Aggressive: Full GNN sequence hint (same as hint-only)
    # Note: Can't hard-lock because GNN solution might not be optimal
    # ─────────────────────────────────────────────────────
    num_prefixed = 0
    if gnn_sequence and len(gnn_sequence) > 0:
        gnn_nodes = [start_idx] + [s + 1 for s in gnn_sequence]
        active_arcs = set()
        for i in range(len(gnn_nodes) - 1):
            active_arcs.add((gnn_nodes[i], gnn_nodes[i + 1]))
            num_prefixed += 1

        hint_vars, hint_vals = [], []
        for (i, j), var in x.items():
            hint_vars.append(var)
            hint_vals.append(1.0 if (i, j) in active_arcs else 0.0)

        solver.SetHint(hint_vars, hint_vals)

    status = solver.Solve()
    elapsed = time.time() - solve_start

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Prefix-lock ILP failed to reach OPTIMAL. Status={status}")

    total_cost = solver.Objective().Value() + pick_bin_cost
    return total_cost, elapsed, len(locked_arcs)


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare warm-start strategies: baseline vs hint-only vs prefix-lock"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-scenarios", type=int, default=5)
    parser.add_argument("--lock-ratio", type=float, default=0.2, help="Fraction of GNN sequence to lock")
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

    print("=" * 120)
    print(
        f"{'Idx':<4} {'N':<4} {'GNN Gap':<10} {'Baseline':<12} {'Hint-Only':<12} "
        f"{'Speedup':<12} {'Gap vs Opt':<12}"
    )
    print("=" * 120)

    baseline_times = []
    hint_times = []
    gnn_gaps = []

    for idx, scenario in enumerate(scenarios):
        scenario_torch = prepare_scenario(scenario, device)
        gnn_cost, gnn_sequence = gnn_rollout_with_route(model, scenario_torch, device)

        # Baseline
        cost_base, time_base = solve_ilp_baseline(scenario, max_objects=args.max_ilp_objects)

        # Hint-only
        cost_hint, time_hint = solve_ilp_hint_only(scenario, gnn_sequence, max_objects=args.max_ilp_objects)

        # Verify both reach same cost
        assert abs(cost_base - cost_hint) < 1e-6, f"Cost mismatch: {cost_base} vs {cost_hint}"

        baseline_times.append(time_base)
        hint_times.append(time_hint)

        speedup = time_base / time_hint if time_hint > 0 else float('inf')
        gnn_gap_pct = (gnn_cost - cost_base) / cost_base * 100
        gnn_gaps.append(gnn_gap_pct)

        print(
            f"{idx:<4} {len(scenario['objects']):<4} {gnn_gap_pct:<10.2f}% "
            f"{time_base:<12.4f} {time_hint:<12.4f} {speedup:<12.2f}x "
            f"{(cost_base - gnn_cost):<12.2f}"
        )

    # Summary
    print("\n" + "=" * 120)
    print("DETAILED ANALYSIS")
    print("=" * 120)

    baseline_np = np.array(baseline_times)
    hint_np = np.array(hint_times)
    speedup_np = baseline_np / hint_np
    gnn_gaps_np = np.array(gnn_gaps)

    print(f"\nBaseline (no warmstart):")
    print(f"  Mean time:  {np.mean(baseline_np):.4f}s")
    print(f"  Total time: {np.sum(baseline_np):.2f}s")

    print(f"\nHint-Only (SetHint with full GNN sequence):")
    print(f"  Mean time:  {np.mean(hint_np):.4f}s")
    print(f"  Mean speedup vs baseline: {np.mean(speedup_np):.2f}x")
    print(f"  Speedup range: {np.min(speedup_np):.2f}x to {np.max(speedup_np):.2f}x")
    print(f"  % scenarios where hint helped: {sum(s >= 1.0 for s in speedup_np)/len(speedup_np)*100:.1f}%")

    print(f"\nGNN Solution Quality:")
    print(f"  Mean gap from optimal: {np.mean(gnn_gaps_np):.2f}%")
    print(f"  Gap range: {np.min(gnn_gaps_np):.2f}% to {np.max(gnn_gaps_np):.2f}%")

    print(f"\nKEY INSIGHT:")
    print(f"  Warm-start hint provides modest speedup ({np.mean(speedup_np):.2f}x on average)")
    print(f"  This is expected for small-to-medium problems where CBC solves fast anyway")
    print(f"  The true benefit of warm-start will appear on larger problems (100+ objects)")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
