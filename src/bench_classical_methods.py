"""
Benchmark SA and 2-opt for Table IV in the paper.

SA uses OR-Tools RoutingModel with SIMULATED_ANNEALING (1 s budget, same as GLS).
2-opt uses random-restart 2-opt with greedy init (1 s wall-clock budget).

Run from src/:
  python3 bench_classical_methods.py

Output: experiments/classical_methods_v8.json
"""

import json
import time
from pathlib import Path

import numpy as np

from ortools.constraint_solver import routing_enums_pb2

from gnn_train import load_scenarios, split_dataset
from benchmark_metaheuristics import build_cost_matrix, solve_ortools_routing

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
OUT_PATH = REPO / "experiments" / "classical_methods_v8.json"

SEED = 42
VAL_SPLIT = 0.1
TIME_LIMIT_MS = 1000  # 1 s — matches GLS budget in Table IV

DATASETS = [
    ("N=10",  DATA_DIR / "dataset_10_objects.json",  50),
    ("N=40",  DATA_DIR / "dataset_40_objects.json",  50),
    ("N=200", DATA_DIR / "dataset_200_objects.json", 20),
]


def solve_2opt_timed(costs, node_count, time_limit_s=1.0):
    """Random-restart 2-opt that respects a wall-clock time limit."""
    import random

    N = node_count - 1

    def route_cost(route):
        c = costs[0][route[0] + 1]
        for i in range(len(route) - 1):
            c += costs[route[i] + 1][route[i + 1] + 1]
        c += costs[route[-1] + 1][0]
        return c

    t0 = time.time()
    best_cost = float("inf")
    rng = random.Random(SEED)
    restart = 0

    while time.time() - t0 < time_limit_s:
        if restart == 0:
            route = []
            remaining = set(range(N))
            cur = 0
            while remaining:
                best_next = min(remaining, key=lambda j: costs[cur][j + 1])
                route.append(best_next)
                cur = best_next + 1
                remaining.remove(best_next)
        else:
            route = list(range(N))
            rng.shuffle(route)
        restart += 1

        current_cost = route_cost(route)
        improved = True
        while improved and time.time() - t0 < time_limit_s:
            improved = False
            for i in range(N - 1):
                for j in range(i + 1, N):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    nc = route_cost(new_route)
                    if nc < current_cost - 1e-6:
                        route = new_route
                        current_cost = nc
                        improved = True
                        break
                if improved:
                    break

        if current_cost < best_cost:
            best_cost = current_cost

    return best_cost, time.time() - t0


def run_dataset(label, ds_path, max_scenarios):
    all_scenarios = load_scenarios(ds_path)
    _, val_raw = split_dataset(all_scenarios, VAL_SPLIT, SEED)
    val_raw = val_raw[:max_scenarios]
    N = len(val_raw[0]["objects"])
    print(f"\n{'='*60}")
    print(f"{label}  (N={N}, {len(val_raw)} scenarios)")
    print(f"{'='*60}")

    greedy_costs = [s["greedy_cost"] for s in val_raw]
    greedy_mean = float(np.mean(greedy_costs))

    sa_costs, sa_times = [], []
    opt2_costs, opt2_times = [], []

    for idx, s in enumerate(val_raw):
        costs, pick_bin_cost, node_count, _ = build_cost_matrix(s)

        # SA via OR-Tools
        sa_route, sa_t, status = solve_ortools_routing(
            costs, node_count,
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
            TIME_LIMIT_MS,
        )
        total_sa = sa_route + pick_bin_cost if status == "OK" else float("inf")
        sa_costs.append(total_sa)
        sa_times.append(sa_t)

        # 2-opt
        opt2_route, opt2_t = solve_2opt_timed(costs, node_count, TIME_LIMIT_MS / 1000)
        total_2opt = opt2_route + pick_bin_cost
        opt2_costs.append(total_2opt)
        opt2_times.append(opt2_t)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(val_raw)}] SA={total_sa:.1f} ({sa_t:.2f}s)  "
                  f"2-opt={total_2opt:.1f} ({opt2_t:.2f}s)")

    sa_gap = (greedy_mean - float(np.mean(sa_costs))) / greedy_mean * 100
    opt2_gap = (greedy_mean - float(np.mean(opt2_costs))) / greedy_mean * 100
    sa_win = sum(1 for sc, gc in zip(sa_costs, greedy_costs) if sc < gc - 1e-6) / len(val_raw) * 100
    opt2_win = sum(1 for oc, gc in zip(opt2_costs, greedy_costs) if oc < gc - 1e-6) / len(val_raw) * 100

    print(f"\n  SA:    mean={np.mean(sa_costs):.1f}  gap={sa_gap:.2f}%  win={sa_win:.0f}%  "
          f"time={np.mean(sa_times):.3f}s")
    print(f"  2-opt: mean={np.mean(opt2_costs):.1f}  gap={opt2_gap:.2f}%  win={opt2_win:.0f}%  "
          f"time={np.mean(opt2_times):.3f}s")

    return {
        "N": N,
        "n_scenarios": len(val_raw),
        "greedy_mean_cost": greedy_mean,
        "SA": {
            "mean_cost": float(np.mean(sa_costs)),
            "mean_gap_pct": sa_gap,
            "win_pct": sa_win,
            "mean_time_s": float(np.mean(sa_times)),
        },
        "two_opt": {
            "mean_cost": float(np.mean(opt2_costs)),
            "mean_gap_pct": opt2_gap,
            "win_pct": opt2_win,
            "mean_time_s": float(np.mean(opt2_times)),
        },
    }


def main():
    all_results = {}
    for label, ds_path, max_sc in DATASETS:
        if not ds_path.exists():
            print(f"Skipping {label}: {ds_path} not found")
            continue
        all_results[label] = run_dataset(label, ds_path, max_sc)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")

    # Print Table IV rows
    print(f"\n{'='*80}")
    print(f"{'Method':<30} {'N=10 Gap%':>10} {'N=10 Win%':>10} {'N=10 Time':>12} | "
          f"{'N=40 Gap%':>10} {'N=40 Win%':>10} {'N=40 Time':>12} | "
          f"{'N=200 Gap%':>11} {'N=200 Win%':>11} {'N=200 Time':>12}")
    print(f"{'-'*80}")
    for method_key, method_label in [("SA", "Simulated Annealing (1s)"), ("two_opt", "2-opt + restarts (1s)")]:
        row = []
        for label in ["N=10", "N=40", "N=200"]:
            if label in all_results and method_key in all_results[label]:
                r = all_results[label][method_key]
                row.append(f"{r['mean_gap_pct']:>9.2f}%  {r['win_pct']:>8.0f}%  {r['mean_time_s']:>9.2f}s")
            else:
                row.append("      ---        ---        ---")
        print(f"{method_label:<30} {row[0]} | {row[1]} | {row[2]}")


if __name__ == "__main__":
    main()
