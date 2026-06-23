"""
Experiment: does torch.compile speed up the per-step rollout?

Compares the same checkpoint with and without torch.compile on N=200 LEAP.
Also reports step count to detect early-termination (sign of incomplete rollouts).

Run from src/:
  /home/pan-navigator/binning_venv/bin/python experiment_compile_at_n200.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from gnn_ilp_circuit import (
    solve_circuit_cold,
    solve_circuit_pruned,
    load_scenarios,
    prepare_scenario,
    _build_step_graph,
)
from gnn_ilp_circuit import prepare_scenario as _ps_circuit
from gnn_gui import load_model
from gnn_train import split_dataset

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "compile_at_n200.json"

CONFIGS = [
    ("thin_h64_eager",       REPO / "models" / "gnn_thin_h64_n40_curr_to_40.pt", False),
    ("thin_h64_compile",     REPO / "models" / "gnn_thin_h64_n40_curr_to_40.pt", True),
    ("n40_h128_eager",       REPO / "models" / "gnn_final_40obj.pt",             False),
    ("n40_h128_compile",     REPO / "models" / "gnn_final_40obj.pt",             True),
]

VAL_SPLIT = 0.1
VAL_SEED = 42
N_SCENARIOS = 20
LEAP_K = 15
N_WARMUP = 15  # extra warmup to let torch.compile finish JIT


def rollout_with_logits_steps(model, scenario, device):
    """Same as gnn_ilp_circuit.gnn_rollout_with_logits but also returns step count."""
    model.eval()
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=device)
    robot = scenario["start"].clone()
    cost = 0.0
    route = []
    logits_all = []
    steps = 0
    with torch.no_grad():
        for step in range(n):
            nf, ei, om = _build_step_graph(objects, bins, types, mask, robot, device)
            logits = model(nf, ei, om)
            logits_all.append(logits.detach().cpu().numpy().tolist())
            action = int(torch.argmax(logits).item())
            if not mask[action]:
                break
            cost += (
                torch.norm(robot - objects[action]).item()
                + torch.norm(objects[action] - bins[types[action]]).item()
            )
            route.append(action)
            robot = bins[types[action]]
            mask[action] = False
            steps += 1
    return cost, route, logits_all, steps


def time_model(scenarios_raw, model, device, label: str):
    # The rollout fn expects tensorised scenarios but the CP-SAT solvers expect
    # the raw dict. Keep both.
    scenarios_t = [_ps_circuit(s, device) for s in scenarios_raw]
    scenarios_pairs = list(zip(scenarios_raw, scenarios_t))
    gnn_times, total_times, leap_costs, opt_costs, steps_done = [], [], [], [], []
    # Warmup (extra so compile finishes)
    for s_raw, s_t in scenarios_pairs[:N_WARMUP]:
        _, gseq, logits, _ = rollout_with_logits_steps(model, s_t, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        solve_circuit_pruned(s_raw, gseq, logits, k_neighbors=LEAP_K,
                             max_objects=len(s_raw["objects"]) + 2)
    # Timed
    for s_raw, s_t in scenarios_pairs:
        t0 = time.time()
        t_g0 = time.time()
        _, gseq, logits, steps = rollout_with_logits_steps(model, s_t, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_g = time.time() - t_g0
        cost, _, _ = solve_circuit_pruned(
            s_raw, gseq, logits, k_neighbors=LEAP_K, max_objects=len(s_raw["objects"]) + 2,
        )
        total = time.time() - t0
        gnn_times.append(t_g * 1000)
        total_times.append(total * 1000)
        leap_costs.append(cost)
        steps_done.append(steps)
        opt, _ = solve_circuit_cold(s_raw, max_objects=len(s_raw["objects"]) + 2)
        opt_costs.append(opt)
    gaps = (np.array(leap_costs) - np.array(opt_costs)) / np.array(opt_costs) * 100
    return {
        "label": label,
        "gnn_ms_mean": float(np.mean(gnn_times)),
        "gnn_ms_median": float(np.median(gnn_times)),
        "total_ms_mean": float(np.mean(total_times)),
        "total_ms_median": float(np.median(total_times)),
        "steps_mean": float(np.mean(steps_done)),
        "steps_min": int(np.min(steps_done)),
        "gap_mean_pct": float(np.mean(gaps)),
        "gap_max_pct": float(np.max(gaps)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, torch={torch.__version__}\n")

    scenarios_full = load_scenarios(REPO / "data" / "dataset_200_objects.json")
    _, val = split_dataset(scenarios_full, VAL_SPLIT, VAL_SEED)
    val = val[:N_SCENARIOS]
    print(f"N=200, {len(val)} val scenarios\n")

    results = []
    for label, model_path, compile_flag in CONFIGS:
        if not model_path.exists():
            print(f"[skip] {label}: missing {model_path}")
            continue
        print(f"=== {label} (compile={compile_flag}) ===")
        model = load_model(device, str(model_path))
        if compile_flag:
            try:
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            except Exception as e:
                print(f"  [warn] torch.compile failed: {e}; falling back to eager")
        r = time_model(val, model, device, label)
        print(f"  GNN: mean {r['gnn_ms_mean']:.1f} ms / median {r['gnn_ms_median']:.1f} ms")
        print(f"  Total: mean {r['total_ms_mean']:.1f} ms / median {r['total_ms_median']:.1f} ms")
        print(f"  Steps: mean {r['steps_mean']:.1f}, min {r['steps_min']}")
        print(f"  Gap: mean {r['gap_mean_pct']:.4f}%, max {r['gap_max_pct']:.4f}%\n")
        results.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
