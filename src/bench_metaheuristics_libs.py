"""
Standard-library metaheuristic/heuristic baselines for PAP/ATSP, to test whether
a *fast, fair* off-the-shelf baseline (not our hand-rolled pure-Python one)
reaches the optimum at N<=200 within ~1 s.

Libraries:
  - elkai (LKH, C-backed)  -> strongest standard TSP/ATSP heuristic
  - python-tsp             -> standard pure-Python SA + local-search (2-opt)

Same seed-42 val scenarios and same asymmetric pick-leg cost matrix as the paper.

Run from src/:
  python3 bench_metaheuristics_libs.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench_metaheuristics_fresh import (
    load_scenarios, split_val, build_cost_matrix, pick_cost,
)

OPT_GAP = {10: 6.90, 40: 4.30, 200: 1.89}
COST_SCALE = 10000


def lkh_solve(M):
    """LKH via elkai on the integer-scaled asymmetric matrix."""
    import elkai
    N1 = M.shape[0]
    intM = [[int(round(M[i, j] * COST_SCALE)) for j in range(N1)] for i in range(N1)]
    tour = elkai.DistanceMatrix(intM).solve_tsp()      # closed, starts/ends at 0
    if tour[-1] == tour[0]:
        tour = tour[:-1]
    k = tour.index(0)
    tour = tour[k:] + tour[:k]                          # rotate to start at node 0
    return pick_cost(M, tour[1:])                       # object order = tour[1:]


def ptsp_sa(M, budget_s):
    from python_tsp.heuristics import solve_tsp_simulated_annealing
    perm, _ = solve_tsp_simulated_annealing(M, max_processing_time=budget_s)
    perm = list(perm)
    k = perm.index(0)
    perm = perm[k:] + perm[:k]
    return pick_cost(M, perm[1:])


def ptsp_2opt(M, budget_s):
    from python_tsp.heuristics import solve_tsp_local_search
    perm, _ = solve_tsp_local_search(M, perturbation_scheme="two_opt",
                                     max_processing_time=budget_s)
    perm = list(perm)
    k = perm.index(0)
    perm = perm[k:] + perm[:k]
    return pick_cost(M, perm[1:])


def run_method(name, fn, scenarios, greedy, takes_budget, budget_s):
    costs, times = [], []
    for s in scenarios:
        M, place, _ = build_cost_matrix(s)
        t0 = time.time()
        try:
            pc = fn(M, budget_s) if takes_budget else fn(M)
        except Exception as e:
            print(f"  [{name}] error: {type(e).__name__}: {e}")
            return None
        times.append((time.time() - t0) * 1000.0)
        costs.append(pc + place)
    costs = np.array(costs)
    gap = (greedy.mean() - costs.mean()) / greedy.mean() * 100.0
    win = float(np.mean(costs < greedy) * 100.0)
    return {"gap_vs_greedy": float(gap), "win_rate": win,
            "mean_time_ms": float(np.mean(times)), "mean_cost": float(costs.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--budget-ms", type=int, default=1000)
    ap.add_argument("--output", default="../experiments/metaheuristics_libs_v9.json")
    args = ap.parse_args()
    budget_s = args.budget_ms / 1000.0
    sizes = [10, 40, 200]
    max_scen = {10: 50, 40: 50, 200: 20}

    methods = [
        ("LKH (elkai)",      lkh_solve, False),
        ("python-tsp SA",    ptsp_sa,   True),
        ("python-tsp 2-opt", ptsp_2opt, True),
    ]

    all_out = {}
    print(f"{'N':>4} {'method':>17} {'gap%':>7} {'opt%':>6} {'win%':>6} {'time ms':>9}")
    print("-" * 56)
    for N in sizes:
        scen = split_val(load_scenarios(Path(args.data_dir) / f"dataset_{N}_objects.json"))[:max_scen[N]]
        greedy = np.array([s["greedy_cost"] for s in scen])
        all_out[str(N)] = {"n_scenarios": len(scen), "methods": {}}
        for name, fn, tb in methods:
            r = run_method(name, fn, scen, greedy, tb, budget_s)
            if r is None:
                continue
            all_out[str(N)]["methods"][name] = r
            print(f"{N:>4} {name:>17} {r['gap_vs_greedy']:>7.2f} "
                  f"{OPT_GAP[N]:>6.2f} {r['win_rate']:>6.0f} {r['mean_time_ms']:>9.1f}")
        print("-" * 56)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
