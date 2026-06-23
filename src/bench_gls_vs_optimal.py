"""
Validate the GLS-vs-optimal claim properly.

Motivation: the v8 numbers compared GLS and the CP-SAT optimum on only 20
scenarios solved once each, and all OR-Tools metaheuristics (GLS/SA/Tabu/
GreedyDescent) returned byte-identical costs -- i.e. the instances were too
easy to separate the methods. This script:

  1. Computes the PROVEN optimum (CP-SAT + AddCircuit, status=OPTIMAL) and GLS
     on the SAME scenarios, over a configurable (larger) sample.
  2. Probes GLS robustness across multiple first-solution starts and time
     budgets (OR-Tools routing GLS has no exposed RNG seed; start strategy and
     budget are the meaningful variation axes).
  3. Reports the per-scenario gap-to-optimum distribution: mean, worst-case,
     p95, and the fraction of instances where GLS fails to reach the optimum.
  4. Cross-references the cached per-scenario LEAP and CP-SAT costs from
     experiments/ilp_timing_circuit_v8.json (first 20 N=200 val scenarios) for
     a true three-way (CP-SAT vs LEAP vs GLS) comparison -- no torch needed.

Self-contained: only depends on ortools + numpy (no torch / torch_geometric),
so it runs without the GNN stack. The CP-SAT formulation is copied verbatim
from gnn_ilp_circuit.py so the optimum matches the paper's numbers (verified
against the cached unpruned costs at runtime).

Run from src/:
    <python-with-ortools> bench_gls_vs_optimal.py --ns 200 40 --n-scenarios 50
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "experiments" / "gls_vs_optimal_v9.json"
CACHED_TIMING = REPO / "experiments" / "ilp_timing_circuit_v8.json"

COST_SCALE = 10000          # matches gnn_ilp_circuit.COST_SCALE
VAL_SPLIT = 0.1             # matches gnn_train.split_dataset
VAL_SEED = 42
CPSAT_MAX_S = 300.0         # exact-solve cap (paper used 300s)


# --------------------------------------------------------------------------
# Data loading / split -- replicated from gnn_train (numpy only, no torch)
# --------------------------------------------------------------------------
def load_scenarios(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def split_dataset(scenarios: List[Dict], val_split: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(scenarios))
    rng.shuffle(indices)
    val_size = int(len(indices) * val_split)
    val_idx = set(indices[:val_size].tolist())
    train, val = [], []
    for i, s in enumerate(scenarios):
        (val if i in val_idx else train).append(s)
    return train, val


# --------------------------------------------------------------------------
# Cost matrix + exact CP-SAT circuit -- copied verbatim from gnn_ilp_circuit
# --------------------------------------------------------------------------
def build_cost_matrix(scenario: Dict):
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    n = len(objects)
    pick_bin_cost = float(np.sum(np.linalg.norm(objects - bins[types], axis=1)))
    node_count = n + 1

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    costs = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for j in range(1, node_count):
        costs[0][j] = dist(start, objects[j - 1])
        costs[j][0] = 0.0
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i != j:
                costs[i][j] = dist(bins[types[i - 1]], objects[j - 1])
    return costs, pick_bin_cost, node_count


def _build_circuit_model(int_costs, node_count):
    model = cp_model.CpModel()
    arcs = []
    arc_vars = {}
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                v = model.NewBoolVar(f"a_{i}_{j}")
                arc_vars[i, j] = v
                arcs.append((i, j, v))
    model.AddCircuit(arcs)
    model.Minimize(sum(int_costs[i][j] * arc_vars[i, j]
                       for i in range(node_count) for j in range(node_count) if i != j))
    return model, arc_vars


def solve_optimal(scenario: Dict) -> Tuple[float, float, bool]:
    """Return (optimal_cost, elapsed_s, proven_optimal)."""
    costs, pbc, node_count = build_cost_matrix(scenario)
    int_costs = [[int(round(costs[i][j] * COST_SCALE)) for j in range(node_count)]
                 for i in range(node_count)]
    model, _ = _build_circuit_model(int_costs, node_count)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = CPSAT_MAX_S
    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed: status={status}")
    cost = solver.ObjectiveValue() / COST_SCALE + pbc
    return cost, elapsed, status == cp_model.OPTIMAL


# --------------------------------------------------------------------------
# GLS via OR-Tools RoutingModel -- copied from benchmark_metaheuristics,
# with the first-solution strategy exposed for robustness probing.
# --------------------------------------------------------------------------
def solve_gls(costs, node_count, pbc, time_limit_ms: int, first_strategy) -> Tuple[float, float]:
    int_costs = [[int(round(c * COST_SCALE)) for c in row] for row in costs]
    manager = pywrapcp.RoutingIndexManager(node_count, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cb(from_index, to_index):
        return int_costs[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    tid = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tid)

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = first_strategy
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(time_limit_ms)

    t0 = time.time()
    sol = routing.SolveWithParameters(p)
    elapsed = time.time() - t0
    if not sol:
        return float("inf"), elapsed
    return sol.ObjectiveValue() / COST_SCALE + pbc, elapsed


FIRST_STRATEGIES = {
    "path_cheapest_arc": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    "savings": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    "global_cheapest_arc": routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
    "local_cheapest_arc": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
    "christofides": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
}


# --------------------------------------------------------------------------
def pct(a, b):
    """(a - b) / b * 100  (gap of a above b)."""
    return (a - b) / b * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[200, 40])
    ap.add_argument("--n-scenarios", type=int, default=50,
                    help="scenarios per N (first k of the seed-42 val split)")
    ap.add_argument("--gls-budgets-ms", type=str, default="1000",
                    help="comma-separated GLS time budgets")
    ap.add_argument("--starts", type=str,
                    default="path_cheapest_arc,savings,global_cheapest_arc",
                    help="comma-separated first-solution strategies")
    args = ap.parse_args()

    budgets = [int(x) for x in args.gls_budgets_ms.split(",")]
    starts = [s.strip() for s in args.starts.split(",")]
    for s in starts:
        if s not in FIRST_STRATEGIES:
            raise SystemExit(f"unknown start strategy '{s}'; choices: {list(FIRST_STRATEGIES)}")

    cached = json.loads(CACHED_TIMING.read_text()) if CACHED_TIMING.exists() else {"per_n": {}}

    results = {
        "config": {"val_split": VAL_SPLIT, "val_seed": VAL_SEED,
                   "n_scenarios": args.n_scenarios, "gls_budgets_ms": budgets,
                   "starts": starts, "cpsat_max_s": CPSAT_MAX_S},
        "per_n": {},
    }

    for n in args.ns:
        path = DATA / f"dataset_{n}_objects.json"
        if not path.exists():
            print(f"[skip] N={n}: {path} missing")
            continue
        scenarios = load_scenarios(path)
        _, val = split_dataset(scenarios, VAL_SPLIT, VAL_SEED)
        subset = val[: args.n_scenarios]
        print(f"\n=== N={n}: {len(subset)} scenarios (of {len(val)} val, {len(scenarios)} total) ===")

        per_scenario = []
        t_start = time.time()
        for si, s in enumerate(subset):
            costs, pbc, node_count = build_cost_matrix(s)
            opt, opt_t, proven = solve_optimal(s)
            greedy = float(s["greedy_cost"])

            # GLS across starts x budgets -> best/mean/worst over all configs
            gls_runs = {}
            for b in budgets:
                for st in starts:
                    c, et = solve_gls(costs, node_count, pbc, b, FIRST_STRATEGIES[st])
                    gls_runs[f"{st}@{b}ms"] = {"cost": c, "time_ms": et * 1000.0,
                                               "gap_to_opt_pct": pct(c, opt)}
            gls_costs = [r["cost"] for r in gls_runs.values()]
            rec = {
                "val_pos": si,
                "optimal": opt, "optimal_time_s": opt_t, "proven_optimal": proven,
                "greedy": greedy,
                "gls_best": min(gls_costs), "gls_worst": max(gls_costs),
                "gls_mean": float(np.mean(gls_costs)),
                "gls_best_gap_pct": pct(min(gls_costs), opt),
                "gls_worst_gap_pct": pct(max(gls_costs), opt),
                "gls_runs": gls_runs,
            }
            per_scenario.append(rec)
            if (si + 1) % 5 == 0 or si == len(subset) - 1:
                el = time.time() - t_start
                print(f"  [{si+1}/{len(subset)}] opt={opt:.1f} ({opt_t:.1f}s,{'OPT' if proven else 'FEAS'}) "
                      f"gls_best_gap={rec['gls_best_gap_pct']:.4f}% worst={rec['gls_worst_gap_pct']:.4f}% "
                      f"({el:.0f}s elapsed)")

        # ---- aggregate GLS-vs-optimum distribution ----
        best_gaps = np.array([r["gls_best_gap_pct"] for r in per_scenario])
        worst_gaps = np.array([r["gls_worst_gap_pct"] for r in per_scenario])
        opt_gaps = np.array([pct(r["greedy"], r["optimal"]) for r in per_scenario])  # how far greedy is below opt? sign
        n_not_opt = int(np.sum(best_gaps > 1e-6))
        agg = {
            "n_scenarios": len(per_scenario),
            "all_proven_optimal": all(r["proven_optimal"] for r in per_scenario),
            "gls_best_gap_pct": {"mean": float(best_gaps.mean()), "std": float(best_gaps.std()),
                                 "max": float(best_gaps.max()), "p95": float(np.quantile(best_gaps, 0.95))},
            "gls_worst_gap_pct": {"mean": float(worst_gaps.mean()), "std": float(worst_gaps.std()),
                                  "max": float(worst_gaps.max()), "p95": float(np.quantile(worst_gaps, 0.95))},
            "frac_gls_suboptimal": n_not_opt / len(per_scenario),
            "frac_gls_gap_gt_0p1pct": float(np.mean(best_gaps > 0.1)),
            "frac_gls_gap_gt_0p5pct": float(np.mean(best_gaps > 0.5)),
            "frac_gls_gap_gt_1pct": float(np.mean(best_gaps > 1.0)),
            "optimal_solve_time_s": {
                "mean": float(np.mean([r["optimal_time_s"] for r in per_scenario])),
                "max": float(np.max([r["optimal_time_s"] for r in per_scenario]))},
        }

        # ---- three-way cross-reference with cached LEAP/CP-SAT (overlap region) ----
        threeway = None
        cn = cached.get("per_n", {}).get(str(n))
        if cn and cn.get("leap") and cn.get("unpruned"):
            leap_costs = cn["leap"]["costs"]
            cp_costs = cn["unpruned"]["costs"]
            m = min(len(leap_costs), len(cp_costs), len(per_scenario))
            rows = []
            cp_match = True
            for i in range(m):
                cached_opt = cp_costs[i]
                my_opt = per_scenario[i]["optimal"]
                if abs(cached_opt - my_opt) > 0.5:   # methodology guard
                    cp_match = False
                leap = leap_costs[i]
                gls_best = per_scenario[i]["gls_best"]
                rows.append({
                    "val_pos": i,
                    "cpsat_optimal": cached_opt,
                    "my_cpsat_optimal": my_opt,
                    "leap": leap, "leap_gap_pct": pct(leap, cached_opt),
                    "gls_best": gls_best, "gls_gap_pct": pct(gls_best, cached_opt),
                    "leap_beats_gls": leap < gls_best,
                })
            leap_g = np.array([r["leap_gap_pct"] for r in rows])
            gls_g = np.array([r["gls_gap_pct"] for r in rows])
            threeway = {
                "n_overlap": m,
                "my_cpsat_reproduces_cached": cp_match,
                "leap_gap_pct": {"mean": float(leap_g.mean()), "max": float(leap_g.max())},
                "gls_gap_pct": {"mean": float(gls_g.mean()), "max": float(gls_g.max())},
                "frac_leap_beats_gls_on_cost": float(np.mean([r["leap_beats_gls"] for r in rows])),
                "rows": rows,
            }

        results["per_n"][str(n)] = {"aggregate": agg, "three_way": threeway,
                                    "per_scenario": per_scenario}

        # ---- print summary ----
        print(f"\n  --- N={n} SUMMARY ({agg['n_scenarios']} scenarios, "
              f"all proven optimal: {agg['all_proven_optimal']}) ---")
        print(f"  GLS best-of-{len(starts)*len(budgets)}-configs gap to optimum:")
        print(f"    mean={agg['gls_best_gap_pct']['mean']:.4f}%  "
              f"p95={agg['gls_best_gap_pct']['p95']:.4f}%  "
              f"MAX={agg['gls_best_gap_pct']['max']:.4f}%")
        print(f"  Fraction of scenarios where GLS != optimum: {agg['frac_gls_suboptimal']*100:.1f}%")
        print(f"    gap>0.1%: {agg['frac_gls_gap_gt_0p1pct']*100:.1f}%  "
              f"gap>0.5%: {agg['frac_gls_gap_gt_0p5pct']*100:.1f}%  "
              f"gap>1%: {agg['frac_gls_gap_gt_1pct']*100:.1f}%")
        print(f"  CP-SAT exact solve time: mean={agg['optimal_solve_time_s']['mean']:.2f}s "
              f"max={agg['optimal_solve_time_s']['max']:.2f}s")
        if threeway:
            print(f"  THREE-WAY (cached LEAP vs GLS on {threeway['n_overlap']} scenarios, "
                  f"my CP-SAT reproduces cached: {threeway['my_cpsat_reproduces_cached']}):")
            print(f"    LEAP gap to opt:  mean={threeway['leap_gap_pct']['mean']:.4f}%  "
                  f"max={threeway['leap_gap_pct']['max']:.4f}%")
            print(f"    GLS  gap to opt:  mean={threeway['gls_gap_pct']['mean']:.4f}%  "
                  f"max={threeway['gls_gap_pct']['max']:.4f}%")
            print(f"    LEAP beats GLS on cost in {threeway['frac_leap_beats_gls_on_cost']*100:.0f}% of scenarios")

        OUT.write_text(json.dumps(results, indent=2))

    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
