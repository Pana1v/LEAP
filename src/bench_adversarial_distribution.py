"""Screen adversarial-clustered distributions: does GLS plateau above the
optimum, opening a gap for learning to fill? Cost model is UNCHANGED (4 corner
bins, type->corner mapping, Euclidean); only object geometry and type labels
vary. No GNN needed -- this decides whether the distribution is worth a retrain.

Type-assignment modes (the deceptive lever):
  uniform     : control -- uniform-random positions + random types (the easy case)
  clustered   : K Gaussian clusters, random types (same-type objects scattered)
  farthest    : clustered + each object typed to its FARTHEST corner (max carry)
  interleaved : clustered + high-frequency spatial type pattern (adjacent objects
                map to different corners -> local tours shuttle between corners)

Run from src/ (ortools + numpy).
"""
import argparse, json
from pathlib import Path
import numpy as np

from bench_gls_vs_optimal import (build_cost_matrix, solve_optimal, solve_gls,
                                  FIRST_STRATEGIES)

BINS = [[10.0, 10.0], [90.0, 10.0], [10.0, 90.0], [90.0, 90.0]]
WS = 100.0


def gen(n, mode, rng, n_clusters=4, cstd=7.0):
    if mode == "uniform":
        objs = rng.uniform(8, WS - 8, size=(n, 2))
        types = rng.integers(0, 4, size=n)
    else:
        centers = rng.uniform(18, WS - 18, size=(n_clusters, 2))
        objs = np.stack([np.clip(centers[i % n_clusters] + rng.normal(0, cstd, 2), 4, WS - 4)
                         for i in range(n)])
        if mode == "clustered":
            types = rng.integers(0, 4, size=n)
        elif mode == "farthest":
            types = np.array([int(np.argmax([np.hypot(*(o - np.array(b))) for b in BINS])) for o in objs])
        elif mode == "interleaved":
            types = np.array([(int(o[0] // 11) + int(o[1] // 11)) % 4 for o in objs])
        else:
            raise SystemExit(f"unknown mode {mode}")
    start = rng.uniform(10, WS - 10, size=2)
    return {"objects": objs.tolist(), "types": [int(t) for t in types],
            "bins": BINS, "start": start.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--scen", type=int, default=30)
    ap.add_argument("--modes", type=str, default="uniform,clustered,farthest,interleaved")
    ap.add_argument("--budgets-ms", type=str, default="100,400,1600")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets_ms.split(",")]
    strat = FIRST_STRATEGIES["path_cheapest_arc"]
    print(f"N={args.n}, {args.scen} scen/mode, GLS budgets={budgets}ms\n")
    print(f"  {'mode':<12}{'opt_ms':>8}{'opt_proven':>11}" +
          "".join(f"{'GLS@'+str(b):>11}" for b in budgets))

    for mode in args.modes.split(","):
        rng = np.random.default_rng(args.seed)
        scens = [gen(args.n, mode, rng) for _ in range(args.scen)]
        opt_costs, opt_times, proven = [], [], 0
        for s in scens:
            c, t, pr = solve_optimal(s)
            opt_costs.append(c); opt_times.append(t); proven += int(pr)
        gls_gaps = {b: [] for b in budgets}
        for s, o in zip(scens, opt_costs):
            costs, pbc, nc = build_cost_matrix(s)
            for b in budgets:
                gc, _ = solve_gls(costs, nc, pbc, b, strat)
                gls_gaps[b].append((gc - o) / o * 100)
        row = f"  {mode:<12}{np.mean(opt_times)*1000:>8.0f}{proven}/{len(scens):>9}"
        for b in budgets:
            row += f"{np.mean(gls_gaps[b]):>10.4f}%"
        print(row, flush=True)


if __name__ == "__main__":
    main()
