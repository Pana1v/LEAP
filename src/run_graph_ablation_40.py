"""
Graph-structure ablation on N=40 (500 train) to confirm cyclic star vs k-NN scaling.
Run from src/:  python3 run_graph_ablation_40.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from gnn_train import (
    load_scenarios, split_dataset, prepare_scenario, _build_step_graph,
    PREFIX_STAGES,
)
from run_ablations import (
    train_ablation, _build_fc_graph, _build_knn_graph,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_40_objects.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "ablations"
SEED = 0
TRAIN_SIZE = 500


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {DATA_PATH}")

    scenarios = load_scenarios(str(DATA_PATH))
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(scenarios))
    train_raw = [scenarios[i] for i in indices[:TRAIN_SIZE]]
    val_raw = [scenarios[i] for i in indices[TRAIN_SIZE:]]

    train_data = [prepare_scenario(s, device) for s in train_raw]
    val_data = [prepare_scenario(s, device) for s in val_raw]
    object_count = train_data[0]["objects"].size(0)
    stages = [p for p in PREFIX_STAGES if p <= object_count] or [object_count]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Objects: {object_count}")
    print(f"Stages: {stages}")
    print(f"Started at: {time.strftime('%H:%M:%S')}")

    results = {}
    run_start = time.time()

    results["A1_cyclic_star_40"] = train_ablation(
        "[1/3] Cyclic Star (ours) N=40", train_data, val_data, device,
        graph_fn=_build_step_graph, stages=stages,
    )

    results["A1_fully_connected_40"] = train_ablation(
        "[2/3] Fully Connected N=40", train_data, val_data, device,
        graph_fn=_build_fc_graph, stages=stages,
    )

    results["A1_knn_40"] = train_ablation(
        "[3/3] k-NN (k=5) N=40", train_data, val_data, device,
        graph_fn=_build_knn_graph, stages=stages,
    )

    total_time = time.time() - run_start
    print(f"\n[{time.strftime('%H:%M:%S')}] Done in {int(total_time//60)}m{int(total_time%60):02d}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "graph_ablation_40obj.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    print(f"\n{'='*70}")
    print(f"{'Variant':<30} {'Gap vs Greedy':>15} {'Win Rate':>10}")
    print(f"{'='*70}")
    for key, m in results.items():
        print(f"{key:<30} {m['mean_gap_vs_greedy']:>+14.2f}% {m['win_rate']:>9.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
