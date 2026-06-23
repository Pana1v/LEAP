"""
Anytime (iso-time) comparison: how long does GLS need to reach LEAP-quality?

For each N, reuse the cached per-scenario CP-SAT optima and LEAP (cost, time)
from experiments/ilp_timing_circuit_v8.json, then run GLS on the SAME scenarios
across a sweep of time budgets. Output the mean GLS gap-to-optimum at each
budget, and the budget at which GLS matches LEAP's gap -- compared to LEAP's
own wall-clock time.

This answers: at LEAP's runtime, is LEAP better than GLS-given-the-same-time?
LEAP wins only if its (time, gap) point lies BELOW the GLS anytime curve.

Only needs ortools + numpy (no torch). Run from src/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench_gls_vs_optimal import (
    load_scenarios, split_dataset, build_cost_matrix, solve_gls,
    FIRST_STRATEGIES, DATA, REPO, VAL_SPLIT, VAL_SEED,
)

CACHED = REPO / "experiments" / "ilp_timing_circuit_v8.json"
OUT = REPO / "experiments" / "gls_budget_sweep_v9.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--budgets-ms", type=str, default="25,50,100,200,400,800,1600")
    ap.add_argument("--start", type=str, default="path_cheapest_arc")
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets_ms.split(",")]
    strat = FIRST_STRATEGIES[args.start]
    cached = json.loads(CACHED.read_text())["per_n"]

    results = {"config": {"budgets_ms": budgets, "start": args.start}, "per_n": {}}

    for n in args.ns:
        cn = cached.get(str(n))
        if not cn or not cn.get("leap") or not cn.get("unpruned"):
            print(f"[skip] N={n}: no cached LEAP/unpruned")
            continue
        opt_costs = cn["unpruned"]["costs"]
        leap_costs = cn["leap"]["costs"]
        leap_time_ms = cn["leap"]["timing"]["mean_ms"]
        k = len(opt_costs)

        scenarios = load_scenarios(DATA / f"dataset_{n}_objects.json")
        _, val = split_dataset(scenarios, VAL_SPLIT, VAL_SEED)
        subset = val[:k]

        leap_gap = float(np.mean([(l - o) / o * 100 for l, o in zip(leap_costs, opt_costs)]))

        print(f"\n=== N={n} ({k} scenarios) ===")
        print(f"  LEAP: gap={leap_gap:.4f}%  time={leap_time_ms:.0f}ms (GPU, cached)")
        print(f"  {'budget_ms':>10}{'GLS_gap%':>11}{'GLS_time_ms':>13}")

        sweep = []
        for b in budgets:
            gaps, times = [], []
            for s, o in zip(subset, opt_costs):
                costs, pbc, nc = build_cost_matrix(s)
                c, et = solve_gls(costs, nc, pbc, b, strat)
                gaps.append((c - o) / o * 100)
                times.append(et * 1000)
            mg = float(np.mean(gaps))
            mt = float(np.mean(times))
            sweep.append({"budget_ms": b, "gls_gap_pct": mg, "gls_time_ms": mt})
            marker = "  <- below LEAP gap" if mg <= leap_gap else ""
            print(f"  {b:>10}{mg:>11.4f}{mt:>13.0f}{marker}")

        # budget at which GLS first reaches LEAP-quality
        reach = next((s for s in sweep if s["gls_gap_pct"] <= leap_gap), None)
        if reach:
            print(f"  -> GLS reaches LEAP's {leap_gap:.4f}% gap by {reach['budget_ms']}ms "
                  f"(actual {reach['gls_time_ms']:.0f}ms); LEAP needs {leap_time_ms:.0f}ms (GPU).")
            verdict = ("LEAP NOT faster-to-quality than GLS"
                       if reach["gls_time_ms"] <= leap_time_ms
                       else "LEAP reaches its quality faster than GLS")
        else:
            print(f"  -> GLS never reaches LEAP's gap within {max(budgets)}ms "
                  f"(LEAP better in this budget range)")
            verdict = "LEAP faster-to-quality than GLS (gap exists)"
        print(f"  VERDICT: {verdict}")

        results["per_n"][str(n)] = {
            "leap_gap_pct": leap_gap, "leap_time_ms": leap_time_ms,
            "sweep": sweep, "verdict": verdict,
        }
        OUT.write_text(json.dumps(results, indent=2))

    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
