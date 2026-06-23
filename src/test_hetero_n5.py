"""
Sanity test for the heterogeneous GNN on N=5.

Trains HeteroGNNPolicy on dataset_5_objects.json and compares mean cost
against (a) the saved homogeneous baseline for N=5 if present, and (b) the
greedy baseline embedded in each scenario.

Expected outcome: heterogeneous variant should at minimum match the
homogeneous model's gap vs greedy on N=5. If it does, the paper's
``heterogeneous graph with typed edges'' framing is empirically grounded.

Run from src/:  python3 test_hetero_n5.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gnn_train import (
    DEFAULT_ACCUMULATE_GRAD,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS_FINAL,
    DEFAULT_EPOCHS_PER_STAGE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LR,
    DEFAULT_VAL_SPLIT,
    DEFAULT_WEIGHT_DECAY,
    MAX_GRAD_NORM,
    PREFIX_STAGES,
    load_scenarios,
    prepare_scenario,
    rollout_model as rollout_homo,
    split_dataset,
    GNNPolicy,
    FEATURE_DIM,
    compute_scenario_loss,
)
from gnn_hetero import (
    HeteroGNNPolicy,
    compute_scenario_loss_hetero,
    rollout_hetero,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_5_objects.json"
OUT_PATH = PROJECT_ROOT / "experiments" / "hetero_n5_comparison.json"
SEED = 0


def train_model(model, train_data, val_data, device, variant_name, hetero=False):
    object_count = train_data[0]["objects"].size(0)
    stages = [p for p in PREFIX_STAGES if p <= object_count] or [object_count]
    stage_epochs = {
        p: (DEFAULT_EPOCHS_FINAL if p == stages[-1] else DEFAULT_EPOCHS_PER_STAGE)
        for p in stages
    }
    total_batches = sum(stage_epochs[p] * len(train_data) for p in stages)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_batches, 1)
    )

    loss_fn = compute_scenario_loss_hetero if hetero else compute_scenario_loss
    rollout_fn = rollout_hetero if hetero else rollout_homo

    print(f"\n[{variant_name}] Stages: {stages}  epochs: {stage_epochs}")
    best_gap = -math.inf
    best_state = None

    for stage_idx, prefix in enumerate(stages):
        n_epochs = stage_epochs[prefix]
        print(f"  Stage {prefix} ({stage_idx+1}/{len(stages)}), {n_epochs} epochs")
        for epoch in range(n_epochs):
            rng = np.random.default_rng(SEED + epoch + prefix)
            indices = rng.permutation(len(train_data))
            losses = []
            accumulated = 0
            optimizer.zero_grad()
            for idx in indices:
                loss = loss_fn(model, train_data[int(idx)], prefix, device)
                if loss is None:
                    continue
                (loss / DEFAULT_ACCUMULATE_GRAD).backward()
                accumulated += 1
                if accumulated % DEFAULT_ACCUMULATE_GRAD == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                losses.append(loss.item())
            if accumulated % DEFAULT_ACCUMULATE_GRAD != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            model.eval()
            with torch.no_grad():
                costs = [rollout_fn(model, s, device) for s in val_data]
            model.train()
            greedy_costs = [s["greedy_cost"] for s in val_data]
            gap = float(
                np.mean(
                    [
                        (g - c) / g * 100.0 if g > 0 else 0.0
                        for g, c in zip(greedy_costs, costs)
                    ]
                )
            )
            win = sum(1 for g, c in zip(greedy_costs, costs) if c < g)
            win_rate = win / len(val_data) * 100
            if gap > best_gap:
                best_gap = gap
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(
                f"    epoch {epoch+1}  loss={np.mean(losses):.4f}  "
                f"gap={gap:+.3f}%  win={win_rate:.1f}%"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_costs = [rollout_fn(model, s, device) for s in val_data]
    greedy_costs = [s["greedy_cost"] for s in val_data]
    ilp5_costs = [
        s["ilp_costs"].get("5") for s in val_data if s["ilp_costs"].get("5") is not None
    ]
    gaps = [
        (g - c) / g * 100.0 if g > 0 else 0.0
        for g, c in zip(greedy_costs, final_costs)
    ]
    win = sum(1 for g, c in zip(greedy_costs, final_costs) if c < g)

    result = {
        "mean_cost": float(np.mean(final_costs)),
        "mean_greedy_cost": float(np.mean(greedy_costs)),
        "mean_gap_vs_greedy": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps)),
        "win_rate": win / len(val_data) * 100,
        "best_gap_during_training": best_gap,
        "n_val": len(val_data),
    }
    if ilp5_costs:
        ilp_gaps = [
            (c - g) / g * 100.0
            for c, g in zip(final_costs[: len(ilp5_costs)], ilp5_costs)
            if g > 0
        ]
        result["mean_gap_vs_ilp"] = float(np.mean(ilp_gaps))
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {DATA_PATH}")

    scenarios = load_scenarios(DATA_PATH)
    train_raw, val_raw = split_dataset(scenarios, DEFAULT_VAL_SPLIT, SEED)
    train_data = [prepare_scenario(s, device) for s in train_raw]
    val_data = [prepare_scenario(s, device) for s in val_raw]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    torch.manual_seed(SEED)
    homo = GNNPolicy(
        FEATURE_DIM, DEFAULT_HIDDEN_DIM, DEFAULT_ATTENTION_HEADS, DEFAULT_DROPOUT
    ).to(device)
    t0 = time.time()
    homo_result = train_model(homo, train_data, val_data, device, "Homogeneous GAT")
    homo_result["train_time_s"] = time.time() - t0
    homo_result["n_params"] = sum(p.numel() for p in homo.parameters())

    torch.manual_seed(SEED)
    hetero = HeteroGNNPolicy(
        DEFAULT_HIDDEN_DIM, DEFAULT_ATTENTION_HEADS, DEFAULT_DROPOUT
    ).to(device)
    _ = hetero(
        *__prime_hetero(hetero, train_data[0], device)
    )  # trigger lazy init before counting params
    t0 = time.time()
    hetero_result = train_model(
        hetero, train_data, val_data, device, "Heterogeneous HeteroConv", hetero=True
    )
    hetero_result["train_time_s"] = time.time() - t0
    hetero_result["n_params"] = sum(p.numel() for p in hetero.parameters())

    comparison = {
        "homogeneous": homo_result,
        "heterogeneous": hetero_result,
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, r in comparison.items():
        print(
            f"{name:>14}: gap={r['mean_gap_vs_greedy']:+.3f}%  "
            f"win={r['win_rate']:.1f}%  params={r['n_params']}  "
            f"time={r['train_time_s']:.1f}s"
        )
    print("=" * 60)
    delta = (
        hetero_result["mean_gap_vs_greedy"] - homo_result["mean_gap_vs_greedy"]
    )
    print(f"Hetero minus homo gap: {delta:+.3f} percentage points")
    if delta >= -0.2:
        print("PASS: heterogeneous at least matches homogeneous within 0.2 pp.")
    else:
        print(
            "WARN: heterogeneous is worse than homogeneous by more than 0.2 pp — "
            "paper should NOT rely on 'heterogeneous' terminology."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


def __prime_hetero(model, scenario, device):
    """Run one forward pass to materialise lazy parameter shapes."""
    from gnn_hetero import build_hetero_step_graph

    mask = torch.ones(scenario["objects"].size(0), device=device, dtype=torch.bool)
    data, om = build_hetero_step_graph(
        scenario["objects"],
        scenario["bins"],
        scenario["types"],
        mask,
        scenario["start"],
        device,
    )
    return data, om


if __name__ == "__main__":
    main()
