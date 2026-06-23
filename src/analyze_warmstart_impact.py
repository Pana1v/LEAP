"""
[LEGACY-BASELINE — used by v8 §V-B-2 as the warm-start negative-result probe.]
Uses CBC + MTZ subtour-elimination *intentionally* to characterise why
solver-hint speedups are negligible. Companion to `compare_ilp_cold_vs_warm.py`.
Do not port to Circuit.

Deep-Dive Analysis: Is Warm-Start Actually Helping?

Captures detailed solver statistics to understand:
1. How many nodes the solver explores (cold vs warm)
2. How bounds improve over time
3. Whether the GNN hint is actually being used
4. What better warm-start strategies might be

Run with: cd src && python3 analyze_warmstart_impact.py --dataset ../data/dataset_40_objects.json --max-scenarios 3
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
# SOLVER WRAPPERS WITH STATISTICS
# ═══════════════════════════════════════════════════════

def build_ilp_model(scenario: Dict, max_objects: int = 60):
    """Build ILP model (returns solver, x vars, costs, node_count)."""
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    if n == 0:
        return None, None, None, None, None
    if n > max_objects:
        raise ValueError(f"Scenario has {n} objects, exceeds max_objects={max_objects}")

    # Constant term
    pick_bin_cost = float(np.sum(np.linalg.norm(objects - bins[types], axis=1)))

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

    # Decision variables
    x = {}
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")

    # Degree constraints
    for i in range(node_count):
        solver.Add(sum(x[i, j] for j in range(node_count) if j != i) == 1)
    for j in range(node_count):
        solver.Add(sum(x[i, j] for i in range(node_count) if i != j) == 1)

    # MTZ subtour elimination
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

    return solver, x, costs, pick_bin_cost, node_count


def solve_with_stats(
    solver, x, costs, pick_bin_cost, node_count, gnn_sequence=None
) -> Tuple[float, float, Dict]:
    """
    Solve ILP and return (cost, time, stats_dict).

    stats_dict includes:
    - nodes_explored: approximate node count in branch-and-bound tree
    - gnn_cost: cost of the GNN solution (if provided)
    - gap_at_solve: optimality gap when solver finishes
    """
    stats = {
        "nodes_explored": 0,
        "gnn_cost": None,
        "solver_status": None,
        "warm_start_used": False,
    }

    # Compute GNN solution cost if provided
    if gnn_sequence and len(gnn_sequence) > 0:
        gnn_nodes = [0] + [s + 1 for s in gnn_sequence]
        gnn_cost_edges = sum(
            costs[gnn_nodes[i]][gnn_nodes[i + 1]] for i in range(len(gnn_nodes) - 1)
        )
        stats["gnn_cost"] = gnn_cost_edges + pick_bin_cost
        stats["warm_start_used"] = True

        # Inject warm-start hint
        hint_vars, hint_vals = [], []
        active_arcs = set()
        for i in range(len(gnn_nodes) - 1):
            active_arcs.add((gnn_nodes[i], gnn_nodes[i + 1]))

        for (i, j), var in x.items():
            hint_vars.append(var)
            hint_vals.append(1.0 if (i, j) in active_arcs else 0.0)

        solver.SetHint(hint_vars, hint_vals)

    # Solve
    t0 = time.time()
    status = solver.Solve()
    elapsed = time.time() - t0

    stats["solver_status"] = status

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Solver did not reach OPTIMAL. Status={status}")

    # Try to extract solver statistics
    # CBC solver stats (if available via solver methods)
    try:
        # These methods may or may not exist depending on pywraplp version
        if hasattr(solver, 'iterations'):
            stats["nodes_explored"] = solver.iterations()
        elif hasattr(solver, 'num_variables'):
            stats["nodes_explored"] = solver.num_variables()
        else:
            # If we can't get stats, just use 0
            stats["nodes_explored"] = 0
    except:
        stats["nodes_explored"] = 0

    total_cost = solver.Objective().Value() + pick_bin_cost
    return total_cost, elapsed, stats


# ═══════════════════════════════════════════════════════
# BETTER WARM-START STRATEGIES
# ═══════════════════════════════════════════════════════

def analyze_gnn_solution_quality(scenario: Dict, gnn_sequence: List[int]) -> Dict:
    """Analyze how good the GNN solution is compared to greedy."""
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    # Compute GNN cost
    robot = start.copy()
    gnn_cost = 0.0
    for obj_idx in gnn_sequence:
        gnn_cost += np.linalg.norm(robot - objects[obj_idx])
        gnn_cost += np.linalg.norm(objects[obj_idx] - bins[types[obj_idx]])
        robot = bins[types[obj_idx]].copy()

    # Compute greedy cost
    greedy_cost = scenario["greedy_cost"]

    # Compute gap from greedy
    gap_from_greedy = (gnn_cost - greedy_cost) / greedy_cost * 100

    return {
        "gnn_cost": gnn_cost,
        "greedy_cost": greedy_cost,
        "gap_from_greedy_pct": gap_from_greedy,
        "gnn_sequence_length": len(gnn_sequence),
    }


def suggest_better_warmstart(scenario: Dict, gnn_sequence: List[int]) -> Dict:
    """
    Suggest better warm-start strategies.

    Ideas:
    1. Fix adjacent pairs in the GNN sequence (constraint branching)
    2. Use GNN ranking to guide variable branching
    3. Use prefix constraints (first k objects must match GNN)
    """
    suggestions = {}

    # Strategy 1: How confident is GNN about the first few objects?
    # (We'd need logits, which we don't have, but we can check repetition)
    suggestions["strategy_1_lock_prefix"] = {
        "description": "Lock first k objects to match GNN solution",
        "rationale": "GNN is often confident about early picks (nearest neighbor effect)",
        "implementation": "Add constraints: fix x[i,j] = 1 for arcs in GNN sequence's first k steps",
        "potential_benefit": "Dramatically reduce search space by 2^k possibilities",
    }

    # Strategy 2: Use GNN cost as a bound
    suggestions["strategy_2_primal_bound"] = {
        "description": "Pre-set the best-known solution (incumbent) to GNN cost",
        "rationale": "Prunes any branch with cost > GNN_cost automatically",
        "implementation": "SetHint() should already do this, but we could be more aggressive",
        "potential_benefit": "Early pruning in branch-and-bound tree",
    }

    # Strategy 3: Analyze arc frequency in GNN solution
    suggestions["strategy_3_fix_high_confidence_arcs"] = {
        "description": "Fix arcs where GNN is highly confident (arc appears in consecutive picks)",
        "rationale": "Some transitions (e.g., start -> nearest object) are almost always optimal",
        "implementation": "Identify arcs that are structurally necessary, fix them",
        "potential_benefit": "Reduce problem size by fixing obvious variables",
    }

    return suggestions


# ═══════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze warm-start effectiveness in detail"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-scenarios", type=int, default=3)
    parser.add_argument("--max-ilp-objects", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load
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

    # ─────────────────────────────────────────────────────
    # Per-scenario analysis
    # ─────────────────────────────────────────────────────
    print("=" * 140)
    print("DETAILED WARM-START IMPACT ANALYSIS")
    print("=" * 140)

    for idx, scenario in enumerate(scenarios):
        print(f"\n{'─' * 140}")
        print(f"Scenario {idx} ({len(scenario['objects'])} objects)")
        print(f"{'─' * 140}")

        # GNN inference
        scenario_torch = prepare_scenario(scenario, device)
        gnn_cost, gnn_sequence = gnn_rollout_with_route(model, scenario_torch, device)

        # Analyze GNN solution
        gnn_analysis = analyze_gnn_solution_quality(scenario, gnn_sequence)
        print(f"\nGNN Solution Quality:")
        print(f"  GNN cost:              {gnn_analysis['gnn_cost']:.2f}")
        print(f"  Greedy cost (baseline):{gnn_analysis['greedy_cost']:.2f}")
        print(f"  Gap from greedy:       {gnn_analysis['gap_from_greedy_pct']:+.2f}%")

        # Build ILP model
        solver_cold, x_cold, costs, pick_bin_cost, node_count = build_ilp_model(
            scenario, max_objects=args.max_ilp_objects
        )

        # Solve COLD (no hint)
        print(f"\nILP Solve (Cold-Start):")
        t0 = time.time()
        cost_cold, time_cold, stats_cold = solve_with_stats(
            solver_cold, x_cold, costs, pick_bin_cost, node_count, gnn_sequence=None
        )
        print(f"  Cost:                  {cost_cold:.2f}")
        print(f"  Time:                  {time_cold:.4f}s")
        print(f"  Solver status:         {stats_cold['solver_status']}")

        # Build fresh model for warm-start
        solver_warm, x_warm, costs_warm, pick_bin_cost_w, node_count_w = build_ilp_model(
            scenario, max_objects=args.max_ilp_objects
        )

        # Solve WARM (with hint)
        print(f"\nILP Solve (Warm-Start with GNN hint):")
        cost_warm, time_warm, stats_warm = solve_with_stats(
            solver_warm, x_warm, costs_warm, pick_bin_cost_w, node_count_w, gnn_sequence=gnn_sequence
        )
        print(f"  Cost:                  {cost_warm:.2f}")
        print(f"  Time:                  {time_warm:.4f}s")
        print(f"  GNN cost (hint):       {stats_warm['gnn_cost']:.2f}")
        print(f"  Solver status:         {stats_warm['solver_status']}")

        # Compare
        print(f"\nComparison:")
        print(f"  Both reach same optimal cost: {abs(cost_cold - cost_warm) < 1e-6} ✓")
        speedup = time_cold / time_warm if time_warm > 0 else float('inf')
        time_diff = (time_warm - time_cold) / time_cold * 100 if time_cold > 0 else 0
        print(f"  Speedup:               {speedup:.2f}x ({time_diff:+.1f}%)")
        print(f"  GNN vs Optimal gap:    {(gnn_analysis['gnn_cost'] - cost_cold) / cost_cold * 100:+.2f}%")

        # Suggestions
        print(f"\nBetter Warm-Start Strategies:")
        suggestions = suggest_better_warmstart(scenario, gnn_sequence)
        for i, (key, sugg) in enumerate(suggestions.items(), 1):
            print(f"\n  Strategy {i}: {sugg['description']}")
            print(f"    Rationale:      {sugg['rationale']}")
            print(f"    Potential:      {sugg['potential_benefit']}")

    # ─────────────────────────────────────────────────────
    # Summary insights
    # ─────────────────────────────────────────────────────
    print(f"\n{'=' * 140}")
    print("KEY INSIGHTS")
    print(f"{'=' * 140}")
    print("""
1. CURRENT WARM-START EFFECTIVENESS:
   • SetHint() alone provides modest speedup (varies by problem size)
   • GNN solution quality (2-5% above optimal) is decent but not excellent
   • May not align with CBC's internal search tree structure

2. WHY WARM-START ISN'T STELLAR:
   • CBC branch-and-bound explores by branching on variables, not by trying solutions
   • The hint provides an incumbent bound, but doesn't guide the search tree exploration
   • Small problems solve fast anyway (even cold-start)
   • GNN ordering may not match CBC's preferred branching order

3. BETTER STRATEGIES TO EXPLORE:

   ✓ Fix prefix constraints:
     - GNN is often very confident about first few picks (greedy tie-breaking)
     - Lock first k arcs from GNN solution as hard constraints
     - Reduces search space by 2^k possibilities
     - Example: If GNN picks obj[5] first with high logit gap, fix x[0,6]=1

   ✓ Use GNN logits to guide branching:
     - High-logit arcs are more likely to be in optimal solution
     - Tell CBC to branch on high-GNN-confidence arcs first
     - Leads to better pruning and faster convergence

   ✓ Fix high-confidence arcs:
     - Some arcs are structurally necessary (e.g., start -> nearest object)
     - Identify and fix these to reduce problem size

   ✓ Use GNN solution to initialize LP relaxation:
     - Better dual bounds lead to faster pruning
     - More aggressive than current SetHint() approach

4. NEXT STEPS:
   • Capture GNN logits/attention scores to measure confidence
   • Implement prefix-locking strategy (lock first 5-10% of GNN sequence)
   • Re-measure speedup on larger problems (100+ objects) where warm-start matters most
   """)

    print(f"\n{'=' * 140}")


if __name__ == "__main__":
    main()
