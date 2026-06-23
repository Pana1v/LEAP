"""
Regenerate Tables I (Gap vs ILP at N=10, N=20) and II (per-sample N=10 gaps)
sourcing ILP costs from CP-SAT `AddCircuit()` rather than the stored MTZ
values. Costs are mathematically invariant (verified separately) but per
user requirement the paper cannot quote MTZ-sourced figures.

Output:
  experiments/gap_tables_circuit_v8.json
  experiments/gap_table_n10_per_sample_circuit_v8.json   (Table II rows)
"""
import json
from pathlib import Path

import numpy as np
import torch

from gnn_ilp_circuit import solve_circuit_cold
from gnn_train import (
    DEFAULT_DROPOUT,
    FEATURE_DIM,
    GNNPolicy,
    load_scenarios,
    prepare_scenario,
    rollout_model,
    split_dataset,
)

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"

VAL_SPLIT = 0.1
VAL_SEED = 42  # canonical training-time val split; see gnn_train.py default


def load_model(path, device, n_objects=10):
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
        hd = int(blob.get("hidden_dim", 128))
        heads = int(blob.get("heads", 4))
        dp = float(blob.get("dropout", DEFAULT_DROPOUT))
    else:
        sd = blob
        hd = sd["convs.0.bias"].shape[0] if "convs.0.bias" in sd else 128
        heads = sd["convs.0.att_src"].shape[1] if "convs.0.att_src" in sd else 4
        dp = DEFAULT_DROPOUT
    m = GNNPolicy(FEATURE_DIM, hd, heads, dp).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


def eval_dataset(n: int, model_path: Path, device: torch.device) -> dict:
    data_path = REPO / "data" / f"dataset_{n}_objects.json"
    scenarios = load_scenarios(data_path)
    _, val_raw = split_dataset(scenarios, VAL_SPLIT, VAL_SEED)
    print(f"\n=== N={n}: val={len(val_raw)} scenarios ===")

    model = load_model(model_path, device, n)

    gnn_costs, ilp_costs_circuit, ilp_costs_stored, greedy_costs = [], [], [], []
    per_sample = []
    with torch.no_grad():
        for idx, s_raw in enumerate(val_raw):
            s = prepare_scenario(s_raw, device)
            gnn_cost = float(rollout_model(model, s, device))
            ilp_cost, _ = solve_circuit_cold(s_raw, max_objects=max(n + 2, 60))
            stored = float(s_raw.get("ilp_costs", {}).get(str(n), float("nan")))
            greedy_cost = float(s_raw["greedy_cost"])
            gnn_costs.append(gnn_cost)
            ilp_costs_circuit.append(float(ilp_cost))
            ilp_costs_stored.append(stored)
            greedy_costs.append(greedy_cost)
            per_sample.append({
                "val_idx": idx,
                "gnn_cost": gnn_cost,
                "ilp_cost_circuit": float(ilp_cost),
                "ilp_cost_stored_mtz": stored,
                "greedy_cost": greedy_cost,
                "gap_vs_ilp_pct": (gnn_cost - float(ilp_cost)) / float(ilp_cost) * 100.0,
                "gap_vs_greedy_pct": (greedy_cost - gnn_cost) / greedy_cost * 100.0,
            })
            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(val_raw)}")

    gnn_c = np.asarray(gnn_costs)
    ilp_c = np.asarray(ilp_costs_circuit)
    stored_c = np.asarray(ilp_costs_stored)
    gre_c = np.asarray(greedy_costs)

    gap_vs_ilp = (gnn_c - ilp_c) / ilp_c * 100.0
    gap_vs_greedy = (gre_c - gnn_c) / gre_c * 100.0
    win = float((gnn_c < gre_c).mean() * 100.0)
    circuit_vs_mtz = (ilp_c - stored_c) / stored_c * 100.0  # ~zero, sanity

    summary = {
        "n": n,
        "n_scenarios": int(len(val_raw)),
        "val_seed": VAL_SEED,
        "val_split": VAL_SPLIT,
        "gnn_mean_cost": float(gnn_c.mean()),
        "ilp_circuit_mean_cost": float(ilp_c.mean()),
        "ilp_stored_mtz_mean_cost": float(stored_c.mean()),
        "greedy_mean_cost": float(gre_c.mean()),
        "circuit_minus_mtz_mean_pct": float(circuit_vs_mtz.mean()),
        "circuit_minus_mtz_max_abs_pct": float(np.max(np.abs(circuit_vs_mtz))),
        "mean_gap_vs_ilp_pct": float(gap_vs_ilp.mean()),
        "std_gap_vs_ilp_pct": float(gap_vs_ilp.std()),
        "median_gap_vs_ilp_pct": float(np.median(gap_vs_ilp)),
        "max_gap_vs_ilp_pct": float(np.max(gap_vs_ilp)),
        "p95_gap_vs_ilp_pct": float(np.percentile(gap_vs_ilp, 95)),
        "mean_gap_vs_greedy_pct": float(gap_vs_greedy.mean()),
        "win_rate_pct": win,
    }
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "per_sample": per_sample}


def pick_table_ii_rows(per_sample: list, k: int = 10) -> list:
    """Pick 10 scenarios evenly spaced by gap-vs-ILP difficulty (matches v8 Table II caption)."""
    sorted_by_gap = sorted(per_sample, key=lambda x: x["gap_vs_ilp_pct"])
    indices = np.linspace(0, len(sorted_by_gap) - 1, k).astype(int)
    return [sorted_by_gap[i] for i in indices]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    EXP.mkdir(parents=True, exist_ok=True)

    results = {}
    for n in (10, 20):
        model_path = REPO / "models" / (
            "gnn_10obj_best.pt" if n == 10 else "gnn_final_40obj.pt"
        )
        if not model_path.exists():
            print(f"[skip] N={n}: model {model_path} not found")
            continue
        r = eval_dataset(n, model_path, device)
        results[str(n)] = r

    # Table II: per-sample rows for N=10
    if "10" in results:
        rows = pick_table_ii_rows(results["10"]["per_sample"], k=10)
        with open(EXP / "gap_table_n10_per_sample_circuit_v8.json", "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nTable II rows:")
        for i, r in enumerate(rows, 1):
            print(f"  {i:2d}  gnn={r['gnn_cost']:.2f}  ilp={r['ilp_cost_circuit']:.2f}  "
                  f"gap={r['gap_vs_ilp_pct']:.2f}%")

    # Aggregate summary
    out_path = EXP / "gap_tables_circuit_v8.json"
    with open(out_path, "w") as f:
        json.dump({n: r["summary"] for n, r in results.items()}, f, indent=2)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
