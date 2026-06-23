"""
Benchmark classical metaheuristics (OR-Tools routing + standalone 2-opt)
against GNN and Greedy baselines on PAP sequencing.

Uses OR-Tools RoutingModel with ATSP cost matrix from gnn_ilp_circuit.py.

Run from src/:
  python3 benchmark_metaheuristics.py --dataset ../data/dataset_40_objects.json
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from gnn_train import (
    FEATURE_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_DROPOUT,
    GNNPolicy,
    load_scenarios,
    prepare_scenario,
    rollout_model,
    split_dataset,
)

COST_SCALE = 10000
SEED = 42
VAL_SPLIT = 0.1


def build_cost_matrix(scenario_raw: Dict) -> Tuple[List[List[float]], float, int, int]:
    """Build ATSP cost matrix. Node 0 = start, nodes 1..N = objects."""
    objects = scenario_raw["objects"]
    types = scenario_raw["types"]
    bins = scenario_raw["bins"]
    start = scenario_raw["start"]
    N = len(objects)
    node_count = N + 1
    start_idx = 0

    pick_bin_cost = 0.0
    for i in range(N):
        pick_bin_cost += math.dist(objects[i], bins[types[i]])

    costs = [[0.0] * node_count for _ in range(node_count)]
    # From start (node 0) to each object
    for j in range(N):
        costs[0][j + 1] = math.dist(start, objects[j])
    # From object i to object j: distance from bin[type[i]] to object[j]
    for i in range(N):
        bin_i = bins[types[i]]
        for j in range(N):
            if i != j:
                costs[i + 1][j + 1] = math.dist(bin_i, objects[j])
    # Return arcs (to start) are free
    for i in range(1, node_count):
        costs[i][0] = 0.0

    return costs, pick_bin_cost, node_count, start_idx


def solve_ortools_routing(
    costs: List[List[float]],
    node_count: int,
    metaheuristic: int,
    time_limit_ms: int,
) -> Tuple[float, float, str]:
    """Solve ATSP using OR-Tools RoutingModel with given metaheuristic."""
    int_costs = [[int(round(c * COST_SCALE)) for c in row] for row in costs]

    manager = pywrapcp.RoutingIndexManager(node_count, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cost_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_costs[from_node][to_node]

    transit_id = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_id)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = metaheuristic
    search_params.time_limit.FromMilliseconds(time_limit_ms)

    t0 = time.time()
    solution = routing.SolveWithParameters(search_params)
    elapsed = time.time() - t0

    if solution:
        route_cost = solution.ObjectiveValue() / COST_SCALE
        return route_cost, elapsed, "OK"
    return float("inf"), elapsed, "NO_SOLUTION"


def solve_2opt(
    costs: List[List[float]],
    node_count: int,
    n_restarts: int = 50,
    max_iterations: int = 1000,
) -> Tuple[float, float]:
    """Random-restart 2-opt for ATSP."""
    N = node_count - 1  # exclude start node

    def route_cost(route):
        c = costs[0][route[0] + 1]  # start to first
        for i in range(len(route) - 1):
            c += costs[route[i] + 1][route[i + 1] + 1]
        c += costs[route[-1] + 1][0]  # last to start (free)
        return c

    t0 = time.time()
    best_cost = float("inf")
    rng = random.Random(SEED)

    for restart in range(n_restarts):
        if restart == 0:
            # First restart: greedy nearest-neighbor from start
            route = []
            remaining = set(range(N))
            cur = 0  # start node
            while remaining:
                best_next = min(remaining, key=lambda j: costs[cur][j + 1])
                route.append(best_next)
                cur = best_next + 1
                remaining.remove(best_next)
        else:
            route = list(range(N))
            rng.shuffle(route)

        current_cost = route_cost(route)
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            for i in range(N - 1):
                for j in range(i + 1, N):
                    new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                    new_cost = route_cost(new_route)
                    if new_cost < current_cost - 1e-6:
                        route = new_route
                        current_cost = new_cost
                        improved = True
                        break
                if improved:
                    break

        if current_cost < best_cost:
            best_cost = current_cost

    elapsed = time.time() - t0
    return best_cost, elapsed


def load_gnn_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    hdim = ckpt.get("hidden_dim", DEFAULT_HIDDEN_DIM)
    heads = ckpt.get("heads", DEFAULT_ATTENTION_HEADS)
    drop = ckpt.get("dropout", DEFAULT_DROPOUT)
    model = GNNPolicy(FEATURE_DIM, hdim, heads, drop)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Metaheuristic benchmarks for PAP")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="GNN checkpoint path")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--time-limits", type=str, default="1000,5000",
                        help="Comma-separated time limits in ms")
    parser.add_argument("--two-opt-restarts", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    time_limits = [int(x) for x in args.time_limits.split(",")]
    device = torch.device(args.device)

    # Load data
    all_scenarios = load_scenarios(Path(args.dataset))
    _, val_raw = split_dataset(all_scenarios, VAL_SPLIT, SEED)
    val_raw = val_raw[: args.max_scenarios]
    N = len(val_raw[0]["objects"])
    print(f"N={N}, {len(val_raw)} validation scenarios")

    # Auto-detect model path
    model_path = args.model
    if model_path is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
        candidates = [f"gnn_{N}obj_best.pt", f"gnn_final_{N}obj.pt"]
        for c in candidates:
            if (models_dir / c).exists():
                model_path = str(models_dir / c)
                break

    # Metaheuristic configs
    metaheuristics = {
        "GLS": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "SA": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        "Tabu": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        "GreedyDescent": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    }

    # Collect results
    results = {
        "N": N,
        "n_scenarios": len(val_raw),
        "methods": {},
    }

    # 1. Greedy baseline (from dataset)
    greedy_costs = [s["greedy_cost"] for s in val_raw]
    results["methods"]["Greedy_NC"] = {
        "mean_cost": float(np.mean(greedy_costs)),
        "std_cost": float(np.std(greedy_costs)),
        "mean_time_ms": 0.0,
    }
    print(f"\nGreedy NC: mean={np.mean(greedy_costs):.1f}")

    # 2. GNN
    if model_path:
        gnn_model = load_gnn_model(model_path, device)
        gnn_costs, gnn_times = [], []
        val_prepared = [prepare_scenario(s, device) for s in val_raw]
        for sp in val_prepared:
            t0 = time.time()
            with torch.no_grad():
                c = rollout_model(gnn_model, sp, device)
            gnn_times.append((time.time() - t0) * 1000)
            gnn_costs.append(c)
        gap = (np.mean(greedy_costs) - np.mean(gnn_costs)) / np.mean(greedy_costs) * 100
        wins = sum(1 for g, c in zip(greedy_costs, gnn_costs) if c < g)
        results["methods"]["GNN"] = {
            "mean_cost": float(np.mean(gnn_costs)),
            "std_cost": float(np.std(gnn_costs)),
            "mean_time_ms": float(np.mean(gnn_times)),
            "gap_vs_greedy": float(gap),
            "win_rate": wins / len(val_raw) * 100,
        }
        print(f"GNN: mean={np.mean(gnn_costs):.1f}, gap={gap:.2f}%, wins={wins}/{len(val_raw)}, time={np.mean(gnn_times):.0f}ms")
        del gnn_model, val_prepared
    else:
        print("No GNN model found, skipping GNN evaluation")

    # 3. OR-Tools metaheuristics
    for mh_name, mh_enum in metaheuristics.items():
        for tl in time_limits:
            label = f"{mh_name}_{tl}ms"
            mh_costs, mh_times = [], []
            for s in val_raw:
                costs, pbc, nc, si = build_cost_matrix(s)
                c, t, status = solve_ortools_routing(costs, nc, mh_enum, tl)
                mh_costs.append(c + pbc)
                mh_times.append(t * 1000)
            gap = (np.mean(greedy_costs) - np.mean(mh_costs)) / np.mean(greedy_costs) * 100
            wins = sum(1 for g, c in zip(greedy_costs, mh_costs) if c < g)
            results["methods"][label] = {
                "mean_cost": float(np.mean(mh_costs)),
                "std_cost": float(np.std(mh_costs)),
                "mean_time_ms": float(np.mean(mh_times)),
                "gap_vs_greedy": float(gap),
                "win_rate": wins / len(val_raw) * 100,
            }
            print(f"{label}: mean={np.mean(mh_costs):.1f}, gap={gap:.2f}%, wins={wins}/{len(val_raw)}, time={np.mean(mh_times):.0f}ms")

    # 4. 2-opt with random restarts
    twoopt_costs, twoopt_times = [], []
    for s in val_raw:
        costs, pbc, nc, si = build_cost_matrix(s)
        c, t = solve_2opt(costs, nc, n_restarts=args.two_opt_restarts)
        twoopt_costs.append(c + pbc)
        twoopt_times.append(t * 1000)
    gap = (np.mean(greedy_costs) - np.mean(twoopt_costs)) / np.mean(greedy_costs) * 100
    wins = sum(1 for g, c in zip(greedy_costs, twoopt_costs) if c < g)
    results["methods"][f"2opt_{args.two_opt_restarts}restarts"] = {
        "mean_cost": float(np.mean(twoopt_costs)),
        "std_cost": float(np.std(twoopt_costs)),
        "mean_time_ms": float(np.mean(twoopt_times)),
        "gap_vs_greedy": float(gap),
        "win_rate": wins / len(val_raw) * 100,
    }
    print(f"2-opt ({args.two_opt_restarts} restarts): mean={np.mean(twoopt_costs):.1f}, gap={gap:.2f}%, wins={wins}/{len(val_raw)}, time={np.mean(twoopt_times):.0f}ms")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<25} {'Mean Cost':>10} {'Gap%':>8} {'Win%':>7} {'Time(ms)':>10}")
    print(f"{'-'*80}")
    for name, m in results["methods"].items():
        gap_str = f"{m.get('gap_vs_greedy', 0.0):+.2f}%" if "gap_vs_greedy" in m else "---"
        wr_str = f"{m.get('win_rate', 0.0):.1f}%" if "win_rate" in m else "---"
        print(f"{name:<25} {m['mean_cost']:>10.1f} {gap_str:>8} {wr_str:>7} {m['mean_time_ms']:>10.0f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
