"""
Re-time OR-Tools GLS (Guided Local Search, 1s budget) on this hardware so
Table IV's metaheuristic row matches the re-measured CP-SAT Circuit + LEAP
rows. Same seed=42 val split, same sample sizes (N=10: 50, N=40: 50, N=200: 20).

Outputs: experiments/gls_circuit_v8.json
"""
import json
import time
from pathlib import Path

import numpy as np
from ortools.constraint_solver import routing_enums_pb2

from benchmark_metaheuristics import build_cost_matrix, solve_ortools_routing
from gnn_train import load_scenarios, split_dataset

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "gls_circuit_v8.json"

SAMPLE = {10: 50, 40: 50, 200: 20}
TIME_LIMIT_MS = 1000
VAL_SPLIT = 0.1
VAL_SEED = 42


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = {"config": {"time_limit_ms": TIME_LIMIT_MS, "val_seed": VAL_SEED}, "per_n": {}}

    for n, k in SAMPLE.items():
        path = REPO / "data" / f"dataset_{n}_objects.json"
        if not path.exists():
            print(f"[skip] N={n}")
            continue
        scenarios = load_scenarios(path)
        _, val = split_dataset(scenarios, VAL_SPLIT, VAL_SEED)
        val = val[:k]
        print(f"\n=== N={n} (n={len(val)}) ===")

        # Greedy baseline costs (from dataset)
        greedy_costs = np.array([s["greedy_cost"] for s in val], dtype=np.float64)

        gls_costs, gls_times = [], []
        for i, s in enumerate(val):
            costs, pbc, nc, si = build_cost_matrix(s)
            c, t, _status = solve_ortools_routing(
                costs, nc,
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
                TIME_LIMIT_MS,
            )
            gls_costs.append(c + pbc)
            gls_times.append(t * 1000.0)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(val)}: mean cost so far {np.mean(gls_costs):.1f}")

        gls_c = np.array(gls_costs, dtype=np.float64)
        gls_t = np.array(gls_times, dtype=np.float64)
        gap = (greedy_costs - gls_c) / greedy_costs * 100.0
        wins = float((gls_c < greedy_costs).mean() * 100.0)

        entry = {
            "n_scenarios": len(val),
            "mean_cost": float(gls_c.mean()),
            "greedy_mean_cost": float(greedy_costs.mean()),
            "gap_vs_greedy_pct_mean": float(gap.mean()),
            "win_rate_pct": wins,
            "time_mean_ms": float(gls_t.mean()),
            "time_median_ms": float(np.median(gls_t)),
            "time_p95_ms": float(np.percentile(gls_t, 95)),
        }
        out["per_n"][str(n)] = entry
        print(f"  GLS: cost={entry['mean_cost']:.1f}, gap={entry['gap_vs_greedy_pct_mean']:.2f}%, "
              f"win={entry['win_rate_pct']:.1f}%, time={entry['time_mean_ms']:.0f}ms")

        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nFinal: {OUT}")


if __name__ == "__main__":
    main()
