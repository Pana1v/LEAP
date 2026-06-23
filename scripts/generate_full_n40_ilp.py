"""
Compute CP-SAT Circuit optimal full-tour ILP for every scenario in
dataset_40_objects.json (and similarly N=20/N=100 if you want, just edit
DATASETS). Adds `ilp_prefixes['40']` and `ilp_costs['40']` to each scenario.

Parallelized across CPU cores. Each worker gets `num_search_workers=2` so
8 workers × 2 threads ≈ saturates a 16-core box.

Run from repo root:
  /home/pan-navigator/binning_venv/bin/python scripts/generate_full_n40_ilp.py
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Worker function — must be importable for spawn / fork pool
def solve_one(scenario: Dict) -> Tuple[float, list]:
    from ortools.sat.python import cp_model
    from gnn_ilp_circuit import build_cost_matrix, _build_circuit_model, _extract_circuit_solution, COST_SCALE

    n = len(scenario["objects"])
    costs, pick_bin_cost, node_count, start_idx = build_cost_matrix(scenario, max_objects=max(n + 2, 60))
    int_costs = [[int(round(costs[i][j] * COST_SCALE)) for j in range(node_count)] for i in range(node_count)]
    model, arc_vars = _build_circuit_model(int_costs, node_count)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers = 2
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed status={status}")
    arcs = _extract_circuit_solution(solver, arc_vars, node_count)
    succ = {i: j for (i, j) in arcs}
    order = []
    cur = start_idx
    for _ in range(node_count):
        nxt = succ.get(cur)
        if nxt is None or nxt == start_idx:
            break
        order.append(nxt - 1)
        cur = nxt
    cost = (solver.ObjectiveValue() / COST_SCALE) + pick_bin_cost
    return cost, order


DATASETS = [
    ("dataset_40_objects.json", 40),
]


def main():
    for fname, N in DATASETS:
        path = REPO / "data" / fname
        print(f"\n=== {fname} (N={N}) ===")
        with open(path) as f:
            scenarios = json.load(f)
        print(f"loaded {len(scenarios)} scenarios")

        # Skip already-computed
        todo_indices = [i for i, s in enumerate(scenarios)
                        if str(N) not in s.get("ilp_costs", {})]
        print(f"need to solve {len(todo_indices)} (rest already have ilp_costs[{N}])")
        if not todo_indices:
            continue

        t0 = time.time()
        with mp.Pool(processes=8) as pool:
            results = pool.map(solve_one, (scenarios[i] for i in todo_indices), chunksize=8)
        elapsed = time.time() - t0
        print(f"solved {len(results)} in {elapsed:.1f}s ({elapsed / len(results) * 1000:.1f} ms/scenario)")

        for idx, (cost, order) in zip(todo_indices, results):
            scenarios[idx].setdefault("ilp_costs", {})[str(N)] = float(cost)
            scenarios[idx].setdefault("ilp_prefixes", {})[str(N)] = order

        # Backup then overwrite
        backup = path.with_suffix(".json.bak_pre_fullN")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
            print(f"backup saved: {backup}")
        with open(path, "w") as f:
            json.dump(scenarios, f, indent=2)
        print(f"updated: {path}")


if __name__ == "__main__":
    main()
