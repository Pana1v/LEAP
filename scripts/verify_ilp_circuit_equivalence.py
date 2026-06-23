"""
Verify that stored MTZ-CBC ilp_costs match CP-SAT Circuit optima.

Both are exact ATSP solvers; costs are expected to agree within numerical
tolerance (the two formulations use different integer scaling). Any disagreement
beyond tolerance indicates a real bug in one of the solvers.

Output: experiments/ilp_mtz_vs_circuit_equivalence.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from gnn_ilp_circuit import solve_circuit_cold

DATA = REPO / "data"
OUT = REPO / "experiments" / "ilp_mtz_vs_circuit_equivalence.json"

# Spot-check sample size per dataset. Same scenarios always (head of file).
N_PER_DATASET = 100
COST_TOL = 1e-2  # Both solvers scale to integers; MTZ uses raw floats, Circuit uses 1e4 scaling.

DATASETS = [
    "dataset_5_objects.json",
    "dataset_10_objects.json",
    "dataset_20_objects.json",
    "dataset_40_objects.json",
    "dataset_100_objects.json",
    "dataset_200_objects.json",
]


def subscenario(scenario: dict, prefix: int) -> dict:
    """Build a sub-scenario with only the first `prefix` objects (matches MTZ generation slicing)."""
    return {
        "objects": scenario["objects"][:prefix],
        "types": scenario["types"][:prefix],
        "bins": scenario["bins"],
        "start": scenario["start"],
    }


def verify_dataset(name: str) -> dict:
    path = DATA / name
    with open(path) as f:
        scenarios = json.load(f)
    sample = scenarios[: min(N_PER_DATASET, len(scenarios))]
    result = {
        "dataset": name,
        "n_checked": 0,
        "prefixes": {},
    }
    print(f"\n=== {name} (checking {len(sample)} scenarios) ===")
    for scenario in sample:
        ilp_costs = scenario.get("ilp_costs", {})
        for prefix_str, mtz_cost in ilp_costs.items():
            prefix = int(prefix_str)
            sub = subscenario(scenario, prefix)
            circuit_cost, _ = solve_circuit_cold(sub, max_objects=prefix + 2)
            delta = abs(circuit_cost - float(mtz_cost))
            pdict = result["prefixes"].setdefault(prefix_str, {
                "n": 0,
                "max_abs_delta": 0.0,
                "max_rel_delta": 0.0,
                "max_delta_scenario_idx": -1,
                "mismatches": [],
            })
            pdict["n"] += 1
            rel = delta / max(abs(float(mtz_cost)), 1e-9)
            if delta > pdict["max_abs_delta"]:
                pdict["max_abs_delta"] = delta
                pdict["max_delta_scenario_idx"] = sample.index(scenario)
            pdict["max_rel_delta"] = max(pdict["max_rel_delta"], rel)
            if delta > COST_TOL:
                pdict["mismatches"].append({
                    "scenario_idx": sample.index(scenario),
                    "mtz_cost": float(mtz_cost),
                    "circuit_cost": circuit_cost,
                    "delta": delta,
                })
        result["n_checked"] += 1
    for p, pdict in sorted(result["prefixes"].items(), key=lambda x: int(x[0])):
        print(f"  prefix={p}: n={pdict['n']}, max_abs_delta={pdict['max_abs_delta']:.4e}, "
              f"max_rel_delta={pdict['max_rel_delta']:.4e}, mismatches={len(pdict['mismatches'])}")
    return result


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {"tol": COST_TOL, "n_per_dataset": N_PER_DATASET, "datasets": []}
    for name in DATASETS:
        results["datasets"].append(verify_dataset(name))
    results["wall_clock_s"] = time.time() - t0

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten: {OUT}")
    print(f"Wall clock: {results['wall_clock_s']:.1f}s")

    # Summary
    any_mismatch = False
    for d in results["datasets"]:
        for p, pdict in d["prefixes"].items():
            if pdict["mismatches"]:
                any_mismatch = True
                print(f"!! {d['dataset']} prefix={p}: {len(pdict['mismatches'])} mismatches")
    if not any_mismatch:
        print("\nAll spot-checked MTZ optima match CP-SAT Circuit optima within tolerance.")


if __name__ == "__main__":
    main()
