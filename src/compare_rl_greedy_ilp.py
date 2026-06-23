"""
[LEGACY — not used by the v8 paper.]
RL-vs-greedy-vs-ILP comparison using CBC + MTZ subtour-elimination.
The RL track was dropped from v8 and the ILP comparison was reworked using
`gnn_ilp_circuit.py`. Kept for historical reference; do not source new
results from it.
"""
import argparse
import json
import math
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ortools.linear_solver import pywraplp
from stable_baselines3 import PPO

from binning_env import BinningEnv

# Configurable constants (avoid magic numbers)
INVALID_COST_MULTIPLIER = 2.0  # fallback multiplier when rollout fails
ILP_MAX_OBJECTS_DEFAULT = 60  # cap ILP size to keep solve time reasonable


def rollout_rl(model: PPO, scenario: Dict, max_objects: int) -> Tuple[float, bool, List[np.ndarray]]:
    """Run a deterministic rollout and return cost, success flag, and path."""
    env = BinningEnv(scenario, max_objects=max_objects)
    obs, _ = env.reset()
    done = False
    path = [env.robot_pos.copy() * env.workspace_size]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        path.append(env.robot_pos.copy() * env.workspace_size)
        done = terminated or truncated

    invalid = info.get("invalid_action", False)
    success = (not truncated) and (not invalid) and len(env.remaining_objects) == 0
    cost = env.total_distance if success else scenario["greedy_cost"] * INVALID_COST_MULTIPLIER
    return cost, success, path


def solve_ilp_route(scenario: Dict, max_objects: int) -> Tuple[Optional[float], Optional[List[int]]]:
    """
    Solve an order optimization ILP:
    Minimize start->first + sum(bin_i->obj_j transitions) + sum(obj_i->bin_i).
    Returns (cost, order of object indices) or (None, None) if skipped due to size.
    """
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    if n == 0:
        return 0.0, []
    if n > max_objects:
        return None, None

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
        costs[j][start_idx] = 0.0  # returning to start is free (terminating edge)

    for i in range(1, node_count):
        obj_i = i - 1
        for j in range(1, node_count):
            if i == j:
                continue
            obj_j = j - 1
            costs[i][j] = dist(bins[types[obj_i]], objects[obj_j])

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return None, None

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

    objective = solver.Sum(costs[i][j] * x[i, j] for i in range(node_count) for j in range(node_count) if i != j)
    solver.Minimize(objective)
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return None, None

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
    return total_cost, order_objects


def pct_delta(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100.0


def plot_scenario(scenario: Dict, rl_path: List[np.ndarray], ilp_order: Optional[List[int]]):
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    plt.figure(figsize=(6, 6))
    plt.scatter(objects[:, 0], objects[:, 1], c="blue", label="objects")
    plt.scatter(bins[:, 0], bins[:, 1], c="red", marker="s", label="bins")
    plt.scatter([start[0]], [start[1]], c="green", marker="^", label="start")

    if rl_path:
        rl_arr = np.array(rl_path)
        plt.plot(rl_arr[:, 0], rl_arr[:, 1], "-o", c="orange", label="RL path", linewidth=1)

    if ilp_order:
        seq = [start]
        for idx in ilp_order:
            obj = objects[idx]
            bin_pos = bins[types[idx]]
            seq.extend([obj, bin_pos])
        ilp_arr = np.array(seq)
        plt.plot(ilp_arr[:, 0], ilp_arr[:, 1], "-o", c="purple", label="ILP path", linewidth=1)

    plt.title("RL vs ILP (greedy cost shown in stats)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare RL vs Greedy vs ILP over scenarios.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained RL model (.zip)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--num-scenarios", type=int, default=None, help="Number of scenarios to evaluate (default: all)")
    parser.add_argument("--max-objects", type=int, default=None, help="Max objects for environment state space (defaults to dataset max)")
    parser.add_argument("--max-ilp-objects", type=int, default=ILP_MAX_OBJECTS_DEFAULT, help="ILP object cap (skip ILP if exceeded)")
    parser.add_argument("--plot-idx", type=int, default=0, help="Scenario index to visualize")
    args = parser.parse_args()

    scenarios = json.load(open(args.dataset, "r"))
    if args.num_scenarios:
        scenarios = scenarios[: args.num_scenarios]

    model = PPO.load(args.model)

    rl_costs, greedy_costs, ilp_costs = [], [], []
    sample_rl_path: List[np.ndarray] = []
    sample_ilp_order: Optional[List[int]] = None
    sample_scenario: Optional[Dict] = None

    if args.max_objects is None:
        inferred_max_objects = max(len(s["objects"]) for s in scenarios)
    else:
        inferred_max_objects = args.max_objects

    for idx, scenario in enumerate(scenarios):
        rl_cost, _, rl_path = rollout_rl(model, scenario, max_objects=inferred_max_objects)
        ilp_cost, ilp_order = solve_ilp_route(scenario, max_objects=args.max_ilp_objects)
        rl_costs.append(rl_cost)
        greedy_costs.append(scenario["greedy_cost"])
        ilp_costs.append(ilp_cost)

        if idx == args.plot_idx:
            sample_rl_path = rl_path
            sample_ilp_order = ilp_order
            sample_scenario = scenario

    rl_costs_np = np.array(rl_costs, dtype=np.float32)
    greedy_np = np.array(greedy_costs, dtype=np.float32)
    ilp_np = np.array([c if c is not None else np.nan for c in ilp_costs], dtype=np.float32)

    rl_vs_greedy = pct_delta(greedy_np, rl_costs_np)
    ilp_mask = ~np.isnan(ilp_np)
    rl_vs_ilp = pct_delta(ilp_np[ilp_mask], rl_costs_np[ilp_mask]) if ilp_mask.any() else np.array([])
    ilp_vs_greedy = pct_delta(greedy_np[ilp_mask], ilp_np[ilp_mask]) if ilp_mask.any() else np.array([])

    print("\n=== Summary ===")
    print(f"Scenarios evaluated: {len(scenarios)}")
    print(f"RL mean cost: {np.mean(rl_costs_np):.3f}")
    print(f"Greedy mean cost: {np.mean(greedy_np):.3f}")
    print(f"ILP mean cost (computed): {np.nanmean(ilp_np):.3f}" if ilp_mask.any() else "ILP skipped (too many objects)")
    print(f"RL vs Greedy mean Δ%: {np.mean(rl_vs_greedy):.2f}")
    if ilp_mask.any():
        print(f"ILP vs Greedy mean Δ%: {np.mean(ilp_vs_greedy):.2f}")
        print(f"RL vs ILP mean Δ%: {np.mean(rl_vs_ilp):.2f}")

    if sample_scenario is not None:
        plot_scenario(sample_scenario, sample_rl_path, sample_ilp_order)


if __name__ == "__main__":
    main()

