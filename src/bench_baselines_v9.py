"""
Two standard-library baselines, run as-is on the same seed-42 val scenarios:

  - OR-Tools 2-opt: RoutingModel GREEDY_DESCENT restricted to the library's
    2-opt operator (use_two_opt only), 1 s cap. Replaces the previous
    self-written random-restart 2-opt.
  - LKH (elkai) tuned: solve_tsp(runs=1) -- the standard single-run config
    (vs the runs=10 default that took ~28.9 s at N=200). Config is recorded.

Outputs:
  experiments/ortools_2opt_v9.json
  experiments/lkh_tuned_v9.json   (includes the elkai config)

Run from src/:
  python3 bench_baselines_v9.py
"""

import json
import time
from pathlib import Path

import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from ortools.util.optional_boolean_pb2 import BOOL_TRUE, BOOL_FALSE
import elkai

from bench_metaheuristics_fresh import load_scenarios, split_val, build_cost_matrix, pick_cost

COST = 10000
OPT_GAP = {10: 6.90, 40: 4.30, 200: 1.89}
LKH_RUNS = 1
SIZES = [10, 40, 200]
MAXSCEN = {10: 50, 40: 50, 200: 20}


def int_matrix(M):
    n = M.shape[0]
    return [[int(round(M[i, j] * COST)) for j in range(n)] for i in range(n)]


def ortools_2opt(M, budget_ms=1000):
    nc = M.shape[0]
    iM = int_matrix(M)
    mgr = pywrapcp.RoutingIndexManager(nc, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    tid = routing.RegisterTransitCallback(lambda a, b: iM[mgr.IndexToNode(a)][mgr.IndexToNode(b)])
    routing.SetArcCostEvaluatorOfAllVehicles(tid)
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    ops = sp.local_search_operators
    for f in [a for a in dir(ops) if a.startswith("use_")]:
        try:
            setattr(ops, f, BOOL_FALSE)
        except Exception:
            pass
    ops.use_two_opt = BOOL_TRUE
    sp.time_limit.FromMilliseconds(budget_ms)
    sol = routing.SolveWithParameters(sp)
    return sol.ObjectiveValue() / COST   # pick-leg cost (return arcs free)


def lkh(M, runs=LKH_RUNS):
    iM = int_matrix(M)
    tour = elkai.DistanceMatrix(iM).solve_tsp(runs=runs)
    if tour[-1] == tour[0]:
        tour = tour[:-1]
    k = tour.index(0)
    tour = tour[k:] + tour[:k]
    return pick_cost(M, tour[1:])


def run(method, scenarios, greedy):
    costs, times = [], []
    for s in scenarios:
        M, place, _ = build_cost_matrix(s)
        t0 = time.time()
        pc = method(M)
        times.append((time.time() - t0) * 1000.0)
        costs.append(pc + place)
    costs = np.array(costs)
    return {
        "mean_cost": float(costs.mean()),
        "gap_vs_greedy": float((greedy.mean() - costs.mean()) / greedy.mean() * 100.0),
        "win_rate": float(np.mean(costs < greedy) * 100.0),
        "mean_time_ms": float(np.mean(times)),
    }


def main():
    out_2opt = {"method": "OR-Tools 2-opt (GREEDY_DESCENT, use_two_opt only, 1s cap)", "per_n": {}}
    out_lkh = {"method": f"LKH via elkai, solve_tsp(runs={LKH_RUNS})",
               "config": f"elkai.DistanceMatrix(int_cost).solve_tsp(runs={LKH_RUNS})", "per_n": {}}
    print(f"{'N':>4} {'method':>14} {'gap%':>7} {'opt%':>6} {'win%':>6} {'time ms':>9}")
    print("-" * 52)
    for N in SIZES:
        scen = split_val(load_scenarios(f"../data/dataset_{N}_objects.json"))[:MAXSCEN[N]]
        greedy = np.array([s["greedy_cost"] for s in scen])
        r2 = run(ortools_2opt, scen, greedy)
        rl = run(lkh, scen, greedy)
        out_2opt["per_n"][str(N)] = {"n_scenarios": len(scen), **r2}
        out_lkh["per_n"][str(N)] = {"n_scenarios": len(scen), **rl}
        print(f"{N:>4} {'OR-Tools 2opt':>14} {r2['gap_vs_greedy']:>7.2f} {OPT_GAP[N]:>6.2f} "
              f"{r2['win_rate']:>6.0f} {r2['mean_time_ms']:>9.1f}")
        print(f"{N:>4} {'LKH runs=1':>14} {rl['gap_vs_greedy']:>7.2f} {OPT_GAP[N]:>6.2f} "
              f"{rl['win_rate']:>6.0f} {rl['mean_time_ms']:>9.1f}")
        print("-" * 52)
    Path("../experiments/ortools_2opt_v9.json").write_text(json.dumps(out_2opt, indent=2))
    Path("../experiments/lkh_tuned_v9.json").write_text(json.dumps(out_lkh, indent=2))
    print("saved ortools_2opt_v9.json + lkh_tuned_v9.json")


if __name__ == "__main__":
    main()
