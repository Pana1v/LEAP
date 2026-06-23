"""
Focused re-run of A1 Fully-Connected and A2 MLP ablations with extended
training budget, to verify whether the -24% failure is a fixed-budget
artefact (model collapses to random baseline) or a permanent structural
limit.

Run from src/:  python3 rerun_fc_mlp_ablation.py
"""

import json
import time
from pathlib import Path

import torch

from run_ablations import (
    DATA_PATH,
    SEED,
    VAL_SPLIT,
    MAX_SCENARIOS,
    _build_fc_graph,
    _build_step_graph,
    MLPPolicy,
    train_ablation,
)
from gnn_train import (
    GNNPolicy,
    FEATURE_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_DROPOUT,
    load_scenarios,
    split_dataset,
    prepare_scenario,
)

PROJECT_ROOT = Path(__file__).parent.parent
OUT_PATH = PROJECT_ROOT / "experiments" / "ablations" / "ablation_rerun_fc_mlp.json"

# Extended budget: ~3x the original epochs per stage
EPOCHS_PER_STAGE_EXT = 9
EPOCHS_FINAL_EXT = 18


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scenarios = load_scenarios(DATA_PATH)
    if MAX_SCENARIOS and len(scenarios) > MAX_SCENARIOS:
        scenarios = scenarios[:MAX_SCENARIOS]
    train_raw, val_raw = split_dataset(scenarios, VAL_SPLIT, SEED)
    train_data = [prepare_scenario(s, device) for s in train_raw]
    val_data = [prepare_scenario(s, device) for s in val_raw]
    object_count = train_data[0]["objects"].size(0)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Objects: {object_count}")
    print(f"Extended budget: {EPOCHS_PER_STAGE_EXT} per stage, {EPOCHS_FINAL_EXT} final")
    print(f"Started at: {time.strftime('%H:%M:%S')}")

    results = {}
    run_start = time.time()

    results["A1_fully_connected_extended"] = train_ablation(
        "A1-FC (extended budget)", train_data, val_data, device,
        graph_fn=_build_fc_graph,
        epochs_per_stage=EPOCHS_PER_STAGE_EXT,
        epochs_final=EPOCHS_FINAL_EXT,
    )

    mlp_model = MLPPolicy(FEATURE_DIM, DEFAULT_HIDDEN_DIM, object_count, DEFAULT_DROPOUT).to(device)
    results["A2_mlp_extended"] = train_ablation(
        "A2-MLP (extended budget)", train_data, val_data, device,
        model=mlp_model, graph_fn=_build_step_graph,
        epochs_per_stage=EPOCHS_PER_STAGE_EXT,
        epochs_final=EPOCHS_FINAL_EXT,
    )

    total = time.time() - run_start
    print(f"\nTotal time: {int(total//60)}m{int(total%60):02d}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT_PATH}")

    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k}: gap={v['mean_gap_vs_greedy']:+.2f}%  win={v['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
