"""Is CP-SAT AddCircuit the right backend for LEAP's PRUNED solve, or can a
generic MIP solver do it faster? Solves the IDENTICAL pruned arc set (same k,
same int_costs) with each available backend and compares wall time + optimum.

- CP-SAT: AddCircuit + GNN hint, num_search_workers=1 (matches solve_circuit_pruned).
- HiGHS / SCIP / CBC: MTZ ATSP over the pruned arcs (pywraplp), single-threaded.
  NOTE: no SetHint for these — SetHint segfaults HiGHS through the ortools
  linear_solver wrapper, and they cannot solve the model regardless.
- Gurobi: requires a license/shared lib (absent here) -> reported as unavailable.

Finding (2026-06-06, RTX PRO 2000 Blackwell + torch venv): CP-SAT solves the
N=200 k=15 pruned model (~3976 arcs) to proven optimality in ~150-200 ms;
HiGHS/SCIP/CBC do not find even a FEASIBLE tour within the time limit. The weak
MTZ LP relaxation flounders on a 200-node ATSP where AddCircuit's lazy subtour
elimination does not. CP-SAT is the necessary backend.

Run from src/ with the torch venv:
    /home/pan-navigator/binning_venv/bin/python bench_pruned_backends.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

from gnn_ilp_circuit import (load_scenarios, prepare_scenario,
                             gnn_rollout_with_logits, select_model_by_count,
                             build_cost_matrix, _rank_outgoing_arcs,
                             _build_active_arcs, COST_SCALE)
from gnn_gui import load_model
from gnn_train import split_dataset

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "pruned_backend_comparison.json"
MTZ_BACKENDS = ["HiGHS", "SCIP", "CBC"]


def build_allowed(scenario, gseq, logits, k, max_objects):
    """Reproduce solve_circuit_pruned's allowed-arc set + int_costs."""
    costs, pick_bin_cost, node_count, start_idx = build_cost_matrix(scenario, max_objects)
    int_costs = [[int(round(costs[i][j] * COST_SCALE)) for j in range(node_count)]
                 for i in range(node_count)]
    rankings = _rank_outgoing_arcs(logits, gseq, node_count, start_idx)
    active_gnn = _build_active_arcs(gseq, start_idx)
    allowed = set(active_gnn)
    for fn in range(node_count):
        if fn in rankings:
            for dn, _ in rankings[fn][:k]:
                if dn != fn:
                    allowed.add((fn, dn))
        else:
            d = sorted(((j, costs[fn][j]) for j in range(node_count) if j != fn), key=lambda x: x[1])
            for dn, _ in d[:k]:
                allowed.add((fn, dn))
    for i in range(1, node_count):
        allowed.add((i, start_idx))
    inc = {}
    for (i, j) in allowed:
        inc[j] = inc.get(j, 0) + 1
    for j in range(node_count):
        if inc.get(j, 0) < k:
            d = sorted(((i, costs[i][j]) for i in range(node_count) if i != j), key=lambda x: x[1])
            for src, _ in d[:k]:
                allowed.add((src, j))
    return int_costs, node_count, start_idx, active_gnn, allowed, pick_bin_cost


def solve_cpsat(int_costs, nc, si, active_gnn, allowed, pbc, tl):
    t0 = time.time()
    model = cp_model.CpModel()
    av, arcs = {}, []
    for (i, j) in allowed:
        v = model.NewBoolVar(f"a_{i}_{j}")
        av[i, j] = v
        arcs.append((i, j, v))
    model.AddCircuit(arcs)
    model.Minimize(sum(int_costs[i][j] * av[i, j] for (i, j) in allowed))
    for (i, j) in active_gnn:
        if (i, j) in av:
            model.AddHint(av[i, j], 1)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_search_workers = 1
    st = s.Solve(model)
    return (s.ObjectiveValue() / COST_SCALE + pbc, (time.time() - t0) * 1000,
            st == cp_model.OPTIMAL)


def solve_mtz(backend, int_costs, nc, si, allowed, pbc, tl):
    t0 = time.time()
    solver = pywraplp.Solver.CreateSolver(backend)
    if solver is None:
        return None, 0.0, False
    solver.SetNumThreads(1)
    solver.SetTimeLimit(int(tl * 1000))
    x = {(i, j): solver.BoolVar(f"x_{i}_{j}") for (i, j) in allowed}
    oa, ia = {i: [] for i in range(nc)}, {i: [] for i in range(nc)}
    for (i, j) in allowed:
        oa[i].append((i, j)); ia[j].append((i, j))
    for i in range(nc):
        solver.Add(sum(x[a] for a in oa[i]) == 1)
        solver.Add(sum(x[a] for a in ia[i]) == 1)
    u = {i: solver.NumVar(0, nc - 1, f"u_{i}") for i in range(nc) if i != si}
    for (i, j) in allowed:
        if i != si and j != si:
            solver.Add(u[i] - u[j] + nc * x[i, j] <= nc - 1)
    solver.Minimize(sum(int_costs[i][j] * x[i, j] for (i, j) in allowed))
    st = solver.Solve()
    proven = st == pywraplp.Solver.OPTIMAL
    feasible = st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    cost = solver.Objective().Value() / COST_SCALE + pbc if feasible else None
    return cost, (time.time() - t0) * 1000, proven


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--scen", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=5.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scen = load_scenarios(REPO / "data" / f"dataset_{args.n}_objects.json")
    _, val = split_dataset(scen, 0.1, 42)
    subset = val[:args.scen]
    model = load_model(device, str(select_model_by_count(args.n)))
    print(f"N={args.n} k={args.k}  {args.scen} scen  time_limit={args.time_limit}s "
          f"(single-threaded)\n")

    rows = {"CP-SAT": {"t": [], "ok": 0}}
    for b in MTZ_BACKENDS:
        rows[b] = {"t": [], "ok": 0, "dcost": []}

    for idx, s in enumerate(subset):
        st = prepare_scenario(s, device)
        _, gseq, logits = gnn_rollout_with_logits(model, st, device)
        mo = len(s["objects"]) + 2
        ic, nc, si, active_gnn, allowed, pbc = build_allowed(s, gseq, logits, args.k, mo)

        ref, dt, pr = solve_cpsat(ic, nc, si, active_gnn, allowed, pbc, args.time_limit)
        rows["CP-SAT"]["t"].append(dt); rows["CP-SAT"]["ok"] += int(pr)
        print(f"  scen {idx}: arcs={len(allowed)} | CP-SAT {dt:6.0f}ms "
              f"({'opt' if pr else 'TL'})", flush=True)
        for b in MTZ_BACKENDS:
            c, dtb, prb = solve_mtz(b, ic, nc, si, allowed, pbc, args.time_limit)
            rows[b]["t"].append(dtb); rows[b]["ok"] += int(prb)
            rows[b]["dcost"].append(None if c is None else c - ref)
            tag = "opt" if prb else ("no feasible soln" if c is None else "TL/feasible")
            print(f"           {b:6} {dtb:6.0f}ms ({tag})", flush=True)

    summary = {"config": vars(args), "per_backend": {}}
    print("\n=== summary (mean) ===")
    for name, d in rows.items():
        entry = {"mean_ms": float(np.mean(d["t"])), "proven_optimal": d["ok"],
                 "n_scen": args.scen}
        print(f"  {name:8} {entry['mean_ms']:7.1f} ms  proven {d['ok']}/{args.scen}")
        summary["per_backend"][name] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
