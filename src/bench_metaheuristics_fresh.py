"""
Fresh, clean-room metaheuristics for PAP/ATSP sequencing (NO OR-Tools).

Independent reimplementation of Simulated Annealing, random-restart 2-opt, and
Guided Local Search, operating directly on the asymmetric pick-leg cost matrix.
Purpose: verify the OR-Tools-derived Table III gap-vs-greedy numbers reproduce
under a completely separate implementation, same 1 s wall-clock budget, same
seed-42 validation scenarios.

PAP cost = sum of pick legs (order-dependent ATSP) + sum of place legs (constant).
We optimise only the pick-leg ATSP; the place-leg constant is added back so the
reported cost is comparable to the dataset's greedy_cost.

Run from src/:
  python3 bench_metaheuristics_fresh.py --only-n 10 --budget-ms 1000
  python3 bench_metaheuristics_fresh.py            # all of N=10,40,200
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

SEED = 42
VAL_SPLIT = 0.1

# Paper Table III gap-vs-greedy (%) for cross-checking.
PAPER_GAP = {
    10:  {"SA": 6.73, "2opt": 6.86, "GLS": 6.87},
    40:  {"SA": 4.27, "2opt": 3.66, "GLS": 4.27},
    200: {"SA": 1.89, "2opt": 1.77, "GLS": 1.88},
}
# Exact-optimum gap-vs-greedy (CP-SAT), as an upper reference.
OPT_GAP = {10: 6.9, 40: 4.3, 200: 1.89}


def load_scenarios(path):
    with open(path) as f:
        return json.load(f)


def split_val(scenarios, val_split=VAL_SPLIT, seed=SEED):
    """Reproduce gnn_train.split_dataset exactly: seeded shuffle, val keeps
    original dataset order."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(scenarios))
    rng.shuffle(indices)
    val_size = int(len(indices) * val_split)
    val_idx = set(indices[:val_size].tolist())
    return [s for i, s in enumerate(scenarios) if i in val_idx]


def build_cost_matrix(scenario):
    """Node 0 = start; nodes 1..N = objects. M[i,j] = pick-leg cost (travel
    from the position *after* serving i, i.e. bin[type[i]], to object j).
    Return arcs (column 0) are free. Returns (M, place_const, N)."""
    objects = scenario["objects"]
    types = scenario["types"]
    bins = scenario["bins"]
    start = scenario["start"]
    N = len(objects)
    M = np.zeros((N + 1, N + 1), dtype=np.float64)
    for j in range(N):
        M[0, j + 1] = math.dist(start, objects[j])
    for i in range(N):
        bi = bins[types[i]]
        for j in range(N):
            if i != j:
                M[i + 1, j + 1] = math.dist(bi, objects[j])
    place_const = sum(math.dist(objects[i], bins[types[i]]) for i in range(N))
    return M, place_const, N


def pick_cost(M, perm):
    """Pick-leg cost of an object-node permutation (nodes 1..N). Return free."""
    p = np.asarray(perm)
    c = M[0, p[0]]
    if p.size > 1:
        c += M[p[:-1], p[1:]].sum()
    return float(c)


def greedy_nn(M, N):
    """Nearest-neighbour construction on the pick-leg matrix from the start."""
    remaining = list(range(1, N + 1))
    cur = 0
    perm = []
    while remaining:
        nxt = min(remaining, key=lambda j: M[cur, j])
        perm.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return perm


def two_opt_local(M, perm, max_sweeps=100000):
    """Best-improvement 2-opt to a local optimum on an asymmetric closed tour
    (0 -> perm -> free return). Vectorised candidate ranking via prefix sums;
    the accepted move's cost is verified by real recomputation so the search is
    provably monotonic regardless of the delta formula."""
    perm = list(perm)
    N = len(perm)
    if N < 3:
        return perm, pick_cost(M, perm)
    cur = pick_cost(M, perm)
    for _ in range(max_sweeps):
        qext = np.array([0] + perm + [0])           # positions 0..N+1 (N+1 -> start)
        Dpos = M[np.ix_(qext, qext)]                # (N+2, N+2)
        K = N + 1
        f = Dpos[np.arange(K), np.arange(1, K + 1)]   # forward arcs, len N+1
        r = Dpos[np.arange(1, K + 1), np.arange(K)]   # same arcs reversed
        Fcum = np.concatenate([[0.0], np.cumsum(f)])  # len N+2, Fcum[k]=sum_{<k}
        Rcum = np.concatenate([[0.0], np.cumsum(r)])
        a = np.arange(1, N + 1)[:, None]
        b = np.arange(1, N + 1)[None, :]
        delta = (Dpos[a - 1, b] + Dpos[a, b + 1]
                 - Dpos[a - 1, a] - Dpos[b, b + 1]
                 + (Rcum[b] - Rcum[a]) - (Fcum[b] - Fcum[a]))
        delta = np.where(b > a, delta, np.inf)
        ai, bi = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[ai, bi] >= -1e-7:
            break
        aa, bb = ai + 1, bi + 1                      # tour positions
        cand = perm[:aa - 1] + perm[aa - 1:bb][::-1] + perm[bb:]
        nc = pick_cost(M, cand)
        if nc < cur - 1e-7:
            perm, cur = cand, nc
        else:
            break
    return perm, cur


def solve_2opt(M, N, budget_s, rng):
    t0 = time.time()
    best_perm, best = None, math.inf
    perm, c = two_opt_local(M, greedy_nn(M, N))      # restart 0: greedy NN
    best, best_perm = c, perm
    while time.time() - t0 < budget_s:
        shuf = list(range(1, N + 1))
        rng.shuffle(shuf)
        perm, c = two_opt_local(M, shuf)
        if c < best:
            best, best_perm = c, perm
    return best, best_perm


def solve_sa(M, N, budget_s, rng):
    """SA over relocate + segment-reverse moves; candidate cost recomputed
    (NumPy) each step; time-fraction geometric cooling."""
    perm = greedy_nn(M, N)
    cur = pick_cost(M, perm)
    best, best_perm = cur, perm[:]
    if N < 3:
        return best, best_perm
    T0 = max(1e-9, (cur / N) * 0.3)
    Tend = T0 * 1e-3
    t0 = time.time()
    it = 0
    while True:
        if (it & 255) == 0:
            elapsed = time.time() - t0
            if elapsed >= budget_s:
                break
            T = T0 * (Tend / T0) ** (elapsed / budget_s)
        it += 1
        if rng.random() < 0.5:                       # relocate
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            node = perm[i]
            rest = perm[:i] + perm[i + 1:]
            cand = rest[:j] + [node] + rest[j:]
        else:                                        # segment reverse (2-opt)
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            cand = perm[:i] + perm[i:j + 1][::-1] + perm[j + 1:]
        nc = pick_cost(M, cand)
        d = nc - cur
        if d < 0 or rng.random() < math.exp(-d / T):
            perm, cur = cand, nc
            if cur < best:
                best, best_perm = cur, perm[:]
    return best, best_perm


def solve_gls(M, N, budget_s, rng, alpha=0.3):
    """Guided Local Search: 2-opt local search on M + lambda*P, penalising the
    max-utility tour arc after each local optimum. Tracks best *real* cost."""
    t0 = time.time()
    P = np.zeros_like(M)
    perm, _ = two_opt_local(M, greedy_nn(M, N))
    best = pick_cost(M, perm)
    best_perm = perm[:]
    lam = alpha * best / max(1, N)
    while time.time() - t0 < budget_s:
        perm, _ = two_opt_local(M + lam * P, perm)
        rc = pick_cost(M, perm)
        if rc < best:
            best, best_perm = rc, perm[:]
        nodes = [0] + perm
        u = np.array(nodes[:-1])
        v = np.array(nodes[1:])
        util = M[u, v] / (1.0 + P[u, v])
        mx = util.max()
        for k in np.where(util >= mx - 1e-12)[0]:
            P[u[k], v[k]] += 1.0
    return best, best_perm


def evaluate(dataset_path, max_scenarios, budget_s):
    scenarios = split_val(load_scenarios(dataset_path))[:max_scenarios]
    N = len(scenarios[0]["objects"])
    rng = random.Random(SEED)

    greedy = np.array([s["greedy_cost"] for s in scenarios])
    methods = {"SA": solve_sa, "2opt": solve_2opt, "GLS": solve_gls}
    out = {"N": N, "n_scenarios": len(scenarios), "budget_ms": int(budget_s * 1000),
           "greedy_mean": float(greedy.mean()), "methods": {}}

    for name, fn in methods.items():
        costs, times = [], []
        for s in scenarios:
            M, place, n = build_cost_matrix(s)
            t0 = time.time()
            pc, _ = fn(M, n, budget_s, rng)
            times.append((time.time() - t0) * 1000.0)
            costs.append(pc + place)
        costs = np.array(costs)
        gap = (greedy.mean() - costs.mean()) / greedy.mean() * 100.0
        win = float(np.mean(costs < greedy) * 100.0)
        out["methods"][name] = {
            "mean_cost": float(costs.mean()),
            "gap_vs_greedy": float(gap),
            "win_rate": win,
            "mean_time_ms": float(np.mean(times)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--only-n", type=int, default=None, help="run a single N")
    ap.add_argument("--budget-ms", type=int, default=1000)
    ap.add_argument("--output", default="../experiments/metaheuristics_fresh_v9.json")
    args = ap.parse_args()

    sizes = [args.only_n] if args.only_n else [10, 40, 200]
    max_scen = {10: 50, 40: 50, 200: 20}
    budget_s = args.budget_ms / 1000.0
    data_dir = Path(args.data_dir)

    all_out = {}
    print(f"{'N':>4} {'method':>6} {'fresh gap%':>11} {'paper gap%':>11} "
          f"{'Δ pp':>7} {'opt gap%':>9} {'win%':>6} {'time ms':>8}")
    print("-" * 72)
    for N in sizes:
        res = evaluate(data_dir / f"dataset_{N}_objects.json", max_scen[N], budget_s)
        all_out[str(N)] = res
        for name in ("SA", "2opt", "GLS"):
            m = res["methods"][name]
            paper = PAPER_GAP.get(N, {}).get(name, float("nan"))
            d = m["gap_vs_greedy"] - paper
            print(f"{N:>4} {name:>6} {m['gap_vs_greedy']:>11.2f} {paper:>11.2f} "
                  f"{d:>+7.2f} {OPT_GAP.get(N, float('nan')):>9.2f} "
                  f"{m['win_rate']:>6.0f} {m['mean_time_ms']:>8.0f}")
        print("-" * 72)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
