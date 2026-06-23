"""
Verify the integrity of greedy_cost / ilp_costs fields in the generated
datasets, and sanity-check the ILP formulation itself.

For every scenario we:
  * Re-simulate greedy_sequence and compare to greedy_cost.
  * For each k in ilp_prefixes, re-simulate the sequence over the first k
    objects and compare to ilp_costs[str(k)].
  * Assert ilp_costs[str(k)] <= greedy_cost_on_first_k (optimality).
  * Validate prefix is a permutation of range(k).

Run from repo root:  python3 scripts/verify_dataset_costs.py
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TOL = 1e-3  # cost tolerance (dataset stores float32; 1mm on a 100-unit workspace)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def simulate(sequence, objects, types, bins, start):
    """Simulate a pick-and-place sequence and return total travel cost."""
    robot = start
    cost = 0.0
    for obj_idx in sequence:
        obj = objects[obj_idx]
        binp = bins[types[obj_idx]]
        cost += dist(robot, obj)       # pick leg
        cost += dist(obj, binp)        # place leg
        robot = binp
    return cost


def greedy_on_prefix(objects, types, bins, start, k):
    """Run greedy on the first k objects only."""
    remaining = set(range(k))
    robot = start
    cost = 0.0
    while remaining:
        best_i = None
        best_c = math.inf
        for i in remaining:
            c = dist(robot, objects[i]) + dist(objects[i], bins[types[i]])
            if c < best_c:
                best_c = c
                best_i = i
        cost += best_c
        robot = bins[types[best_i]]
        remaining.remove(best_i)
    return cost


def audit(path):
    with open(path) as f:
        scenarios = json.load(f)

    n_scen = len(scenarios)
    errors = {
        "greedy_mismatch": 0,
        "greedy_seq_not_perm": 0,
        "ilp_mismatch": {},        # by k
        "ilp_not_perm": {},
        "ilp_worse_than_greedy": {},
        "ilp_negative_slack": {},  # ILP > greedy by > TOL (optimality violation)
    }
    worst_greedy = 0.0
    worst_ilp = {}

    for idx, s in enumerate(scenarios):
        objs = s["objects"]
        types = s["types"]
        bins = s["bins"]
        start = s["start"]
        N = len(objs)

        # Greedy
        gseq = s["greedy_sequence"]
        if sorted(gseq) != list(range(N)):
            errors["greedy_seq_not_perm"] += 1
        else:
            c = simulate(gseq, objs, types, bins, start)
            diff = abs(c - s["greedy_cost"])
            if diff > TOL:
                errors["greedy_mismatch"] += 1
            worst_greedy = max(worst_greedy, diff)

        # ILP prefixes
        for k_str, seq in s.get("ilp_prefixes", {}).items():
            k = int(k_str)
            if seq is None:
                continue
            if sorted(seq) != list(range(k)):
                errors["ilp_not_perm"].setdefault(k_str, 0)
                errors["ilp_not_perm"][k_str] += 1
                continue
            c = simulate(seq, objs, types, bins, start)
            stored = s["ilp_costs"][k_str]
            diff = abs(c - stored)
            if diff > TOL:
                errors["ilp_mismatch"].setdefault(k_str, 0)
                errors["ilp_mismatch"][k_str] += 1
            worst_ilp[k_str] = max(worst_ilp.get(k_str, 0.0), diff)

            # Optimality: ILP <= greedy on first k objects
            g = greedy_on_prefix(objs, types, bins, start, k)
            if c > g + TOL:
                errors["ilp_worse_than_greedy"].setdefault(k_str, 0)
                errors["ilp_worse_than_greedy"][k_str] += 1
            if stored > g + TOL:
                errors["ilp_negative_slack"].setdefault(k_str, 0)
                errors["ilp_negative_slack"][k_str] += 1

    return n_scen, errors, worst_greedy, worst_ilp


def main():
    files = sorted(DATA_DIR.glob("dataset_*_objects*.json"))
    print(f"Found {len(files)} datasets in {DATA_DIR}\n")

    for f in files:
        print(f"--- {f.name} ---")
        try:
            n, err, wg, wilp = audit(f)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        status = "OK" if (
            err["greedy_mismatch"] == 0
            and err["greedy_seq_not_perm"] == 0
            and not err["ilp_mismatch"]
            and not err["ilp_not_perm"]
            and not err["ilp_negative_slack"]
        ) else "FAIL"
        print(f"  scenarios:             {n}")
        print(f"  greedy cost mismatches: {err['greedy_mismatch']}  "
              f"(worst |diff| = {wg:.6f})")
        print(f"  greedy seq not perm:    {err['greedy_seq_not_perm']}")
        if err["ilp_mismatch"] or wilp:
            for k in sorted(set(list(err['ilp_mismatch'].keys()) + list(wilp.keys())), key=int):
                print(f"  ilp k={k}: mismatches={err['ilp_mismatch'].get(k,0)}  "
                      f"worst |diff|={wilp.get(k,0.0):.6f}")
        if err["ilp_not_perm"]:
            print(f"  ilp seq not perm: {err['ilp_not_perm']}")
        if err["ilp_worse_than_greedy"]:
            print(f"  ilp_seq_cost > greedy_prefix: {err['ilp_worse_than_greedy']}  "
                  "(simulated ILP seq cost; OK if small slack from precision)")
        if err["ilp_negative_slack"]:
            print(f"  *** STORED ilp_cost > greedy_prefix by > TOL: "
                  f"{err['ilp_negative_slack']}  (optimality violated!)")
        print(f"  {status}\n")


if __name__ == "__main__":
    main()
