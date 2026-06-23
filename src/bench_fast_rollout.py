"""Direct head-to-head benchmark of the original vs fast rollout at N=200.
Measures GNN inference time only (no CP-SAT) and verifies route equality."""
import time

import numpy as np
import torch

from gnn_ilp_circuit import (
    gnn_rollout_with_logits,
    load_scenarios,
    prepare_scenario,
)
from gnn_gui import load_model
from gnn_train import split_dataset
from fast_rollout import fast_rollout_with_logits

from pathlib import Path
REPO = Path(__file__).resolve().parent.parent

VAL_SPLIT = 0.1
VAL_SEED = 42
N_SCENARIOS = 20
N_WARMUP = 10

MODELS = [
    ("n40_h128",     REPO / "models" / "gnn_final_40obj.pt"),
    ("thin_h64_n40", REPO / "models" / "gnn_thin_h64_n40_curr_to_40.pt"),
    ("n200_h128",    REPO / "models" / "gnn_final_200obj.pt"),
]


def bench(scenarios_t, model, device, fn):
    # Warmup
    for s in scenarios_t[:N_WARMUP]:
        fn(model, s, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
    # Timed
    times = []
    routes = []
    for s in scenarios_t:
        t0 = time.time()
        _, route, _ = fn(model, s, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        routes.append(route)
    return np.array(times), routes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc_raw = load_scenarios(REPO / "data" / "dataset_200_objects.json")
    _, val = split_dataset(sc_raw, VAL_SPLIT, VAL_SEED)
    val = val[:N_SCENARIOS]
    scenarios_t = [prepare_scenario(s, device) for s in val]
    print(f"Device: {device}, N=200, {len(scenarios_t)} val scenarios\n")
    print(f"{'Model':<15} {'Variant':<6} {'mean ms':>9} {'med ms':>9}  routes match?")
    print("-" * 60)
    for label, path in MODELS:
        if not path.exists():
            print(f"{label}: missing"); continue
        model = load_model(device, str(path))
        t_orig, r_orig = bench(scenarios_t, model, device, gnn_rollout_with_logits)
        t_fast, r_fast = bench(scenarios_t, model, device, fast_rollout_with_logits)
        match = all(a == b for a, b in zip(r_orig, r_fast))
        print(f"{label:<15} {'orig':<6} {t_orig.mean():>9.1f} {np.median(t_orig):>9.1f}")
        print(f"{label:<15} {'fast':<6} {t_fast.mean():>9.1f} {np.median(t_fast):>9.1f}  {match}")
        print(f"{'':>15} {'->':<6} {t_orig.mean()/t_fast.mean():>9.2f}x speedup\n")


if __name__ == "__main__":
    main()
