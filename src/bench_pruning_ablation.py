"""Publication ablation: is the GNN's LEARNED arc selection necessary, or do
non-learned pruning heuristics work just as well to reduce the exact solver?

Three ways to build the top-k arc set handed to CP-SAT AddCircuit, same k:
  1. cost-kNN          : k cheapest outgoing arcs per node, no warm-start.
  2. cost-kNN + greedy : same arcs + the greedy (nearest-cycle) tour arcs & hint.
  3. GNN (LEAP)        : GNN-ranked top-k arcs + GNN tour arcs & hint.

For each, build the exact pruned model, solve with STATUS exposed, and report
feasibility, gap-to-optimum, and solve time. Pure cost-kNN is reported honestly
(it is usually INFEASIBLE -- the k-nearest-arc graph has no Hamiltonian cycle --
so it forces a fallback to the full exact solve).

ortools + torch + torch_geometric. Run from src/ with the torch venv.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

from gnn_ilp_circuit import (build_cost_matrix, _rank_outgoing_arcs, _build_active_arcs,
                             select_model_by_count, load_scenarios, prepare_scenario,
                             gnn_rollout_with_logits, COST_SCALE)
from gnn_gui import load_model
from gnn_train import split_dataset
import torch

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CACHED = REPO / "experiments" / "ilp_timing_circuit_v8.json"
OUT = REPO / "experiments" / "pruning_ablation_v9.json"
VAL_SPLIT, VAL_SEED, LEAP_K = 0.1, 42, 15
ST = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
      cp_model.INFEASIBLE: "INFEASIBLE", cp_model.UNKNOWN: "UNKNOWN", cp_model.MODEL_INVALID: "INVALID"}


def build_allowed(costs, nc, start_idx, rankings, hint_seq, k):
    active = _build_active_arcs(hint_seq, start_idx) if hint_seq else set()
    allowed = set(a for a in active if a[0] != a[1])  # drop self-loops
    for fn in range(nc):
        if rankings and fn in rankings:
            for dest, _ in rankings[fn][:k]:
                if dest != fn:
                    allowed.add((fn, dest))
        else:
            dd = sorted(((j, costs[fn][j]) for j in range(nc) if j != fn), key=lambda x: x[1])
            for d, _ in dd[:k]:
                allowed.add((fn, d))
    for i in range(1, nc):
        allowed.add((i, start_idx))
    inc = {}
    for (_, j) in allowed:
        inc[j] = inc.get(j, 0) + 1
    for j in range(nc):
        if inc.get(j, 0) < k:
            dd = sorted(((i, costs[i][j]) for i in range(nc) if i != j), key=lambda x: x[1])
            for src, _ in dd[:k]:
                allowed.add((src, j))
    return allowed, active


def solve_pruned(scenario, rankings, hint_seq, k, cap_s):
    costs, pbc, nc, start_idx = build_cost_matrix(scenario, max_objects=len(scenario["objects"]) + 2)
    int_costs = [[int(round(costs[i][j] * COST_SCALE)) for j in range(nc)] for i in range(nc)]
    allowed, active = build_allowed(costs, nc, start_idx, rankings, hint_seq, k)
    m = cp_model.CpModel()
    av = {(i, j): m.NewBoolVar(f"a_{i}_{j}") for (i, j) in allowed}
    m.AddCircuit([(i, j, av[i, j]) for (i, j) in allowed])
    m.Minimize(sum(int_costs[i][j] * av[i, j] for (i, j) in allowed))
    for (i, j) in active:
        if (i, j) in av:
            m.AddHint(av[i, j], 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cap_s
    t0 = time.time()
    st = solver.Solve(m)
    et = time.time() - t0
    feasible = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    cost = (solver.ObjectiveValue() / COST_SCALE + pbc) if feasible else None
    return ST.get(st, st), cost, et * 1000, len(allowed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--k", type=int, default=LEAP_K)
    ap.add_argument("--cap-s", type=float, default=60.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    cached = json.loads(CACHED.read_text())["per_n"]
    results = {"config": {"k": args.k, "cap_s": args.cap_s, "device": str(device)}, "per_n": {}}

    for n in args.ns:
        opt = cached[str(n)]["unpruned"]["costs"]
        scen = split_dataset(load_scenarios(DATA / f"dataset_{n}_objects.json"), VAL_SPLIT, VAL_SEED)[1][:len(opt)]
        model = load_model(device, str(select_model_by_count(n)))
        print(f"\n=== N={n} (k={args.k}, {len(opt)} scen.) ===", flush=True)
        print(f"  {'method':<22}{'feasible':>9}{'gap%':>9}{'worst%':>9}{'solve_ms':>10}{'arcs':>7}", flush=True)

        rows = {}
        for label in ["cost-kNN", "cost-kNN+greedy", "GNN (LEAP)"]:
            gaps, times, arcs, nfeas = [], [], [], 0
            for s, o in zip(scen, opt):
                if label == "cost-kNN":
                    rk, hint = None, None
                elif label == "cost-kNN+greedy":
                    rk, hint = None, s["greedy_sequence"]
                else:
                    st = prepare_scenario(s, device)
                    _, seq, lg = gnn_rollout_with_logits(model, st, device)
                    costs_tmp, _, nc, si = build_cost_matrix(s, max_objects=len(s["objects"]) + 2)
                    rk, hint = _rank_outgoing_arcs(lg, seq, nc, si), seq
                status, cost, ms, na = solve_pruned(s, rk, hint, args.k, args.cap_s)
                times.append(ms); arcs.append(na)
                if cost is not None:
                    nfeas += 1
                    gaps.append((cost - o) / o * 100)
            feas_str = f"{nfeas}/{len(opt)}"
            g = f"{np.mean(gaps):.4f}" if gaps else "  --"
            w = f"{np.max(gaps):.4f}" if gaps else "  --"
            print(f"  {label:<22}{feas_str:>9}{g:>9}{w:>9}{np.mean(times):>10.1f}{np.mean(arcs):>7.0f}", flush=True)
            rows[label] = {"feasible": nfeas, "n": len(opt),
                           "mean_gap_pct": float(np.mean(gaps)) if gaps else None,
                           "worst_gap_pct": float(np.max(gaps)) if gaps else None,
                           "mean_solve_ms": float(np.mean(times)), "mean_arcs": float(np.mean(arcs))}
        results["per_n"][str(n)] = rows
        OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
