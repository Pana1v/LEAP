"""
Ablation study for the paper. Trains 8 variants on N=40 dataset,
evaluates each on the same val set, and saves results to JSON.

Run from src/:  python3 run_ablations.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from tqdm import tqdm

from gnn_train import (
    GNNPolicy,
    FEATURE_DIM,
    NUM_BINS,
    WORKSPACE_SIZE,
    PREFIX_STAGES,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_DROPOUT,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_ACCUMULATE_GRAD,
    DEFAULT_EPOCHS_PER_STAGE,
    DEFAULT_EPOCHS_FINAL,
    MAX_GRAD_NORM,
    MASK_FILL_VALUE,
    load_scenarios,
    split_dataset,
    prepare_scenario,
    rollout_model,
    evaluate,
    compute_scenario_loss,
    _build_step_graph,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_40_objects.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "ablations"
SEED = 0
VAL_SPLIT = 0.1
MAX_SCENARIOS = 1000  # Cap total scenarios to keep compute manageable on N=40


# ═══════════════════════════════════════════════════════
# ALTERNATIVE GRAPH BUILDERS (A1)
# ═══════════════════════════════════════════════════════

def _build_fc_graph(objects, bins, types, mask, robot_world, device):
    """Fully connected graph — edges between all node pairs."""
    n_objects = objects.size(0)
    bin_offset = 1 + n_objects
    total = 1 + n_objects + NUM_BINS

    # Node features (same as star)
    robot_norm = robot_world / WORKSPACE_SIZE
    obj_norm = objects / WORKSPACE_SIZE
    bin_norm = bins / WORKSPACE_SIZE
    type_one_hot = F.one_hot(types, num_classes=NUM_BINS).float()

    robot_feat = torch.zeros((1, FEATURE_DIM), device=device)
    robot_feat[0, 0:2] = robot_norm

    object_feat = torch.zeros((n_objects, FEATURE_DIM), device=device)
    object_feat[:, 0:2] = obj_norm
    object_feat[:, 2:4] = bin_norm[types]
    object_feat[:, 4:4 + NUM_BINS] = type_one_hot
    object_feat[:, 4 + NUM_BINS] = mask.float()

    bin_feat = torch.zeros((NUM_BINS, FEATURE_DIM), device=device)
    bin_feat[:, 0:2] = bin_norm
    bin_feat[:, 2:4] = bin_norm
    bin_feat[:, 4:4 + NUM_BINS] = torch.eye(NUM_BINS, device=device)

    node_features = torch.cat([robot_feat, object_feat, bin_feat], dim=0)

    # Fully connected edges (all pairs, excluding self-loops)
    src, dst = [], []
    for i in range(total):
        for j in range(total):
            if i != j:
                src.append(i)
                dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return node_features, edge_index, mask


def _build_knn_graph(objects, bins, types, mask, robot_world, device, k=5):
    """k-NN graph — each node connects to k spatially nearest nodes + bin edges."""
    n_objects = objects.size(0)
    bin_offset = 1 + n_objects

    robot_norm = robot_world / WORKSPACE_SIZE
    obj_norm = objects / WORKSPACE_SIZE
    bin_norm = bins / WORKSPACE_SIZE
    type_one_hot = F.one_hot(types, num_classes=NUM_BINS).float()

    robot_feat = torch.zeros((1, FEATURE_DIM), device=device)
    robot_feat[0, 0:2] = robot_norm

    object_feat = torch.zeros((n_objects, FEATURE_DIM), device=device)
    object_feat[:, 0:2] = obj_norm
    object_feat[:, 2:4] = bin_norm[types]
    object_feat[:, 4:4 + NUM_BINS] = type_one_hot
    object_feat[:, 4 + NUM_BINS] = mask.float()

    bin_feat = torch.zeros((NUM_BINS, FEATURE_DIM), device=device)
    bin_feat[:, 0:2] = bin_norm
    bin_feat[:, 2:4] = bin_norm
    bin_feat[:, 4:4 + NUM_BINS] = torch.eye(NUM_BINS, device=device)

    node_features = torch.cat([robot_feat, object_feat, bin_feat], dim=0)

    # Collect positions for all nodes
    all_pos = torch.cat([
        robot_world.unsqueeze(0),
        objects,
        bins,
    ], dim=0)

    total = all_pos.size(0)
    src, dst = [], []

    for i in range(total):
        dists = torch.norm(all_pos - all_pos[i], dim=1)
        dists[i] = float('inf')
        _, nn_idx = torch.topk(dists, min(k, total - 1), largest=False)
        for j in nn_idx:
            src.append(i)
            dst.append(j.item())

    # Also add object→bin edges for valid objects (ensure type info flows)
    for obj_idx in torch.nonzero(mask, as_tuple=False).flatten():
        obj_node = 1 + obj_idx.item()
        bin_node = bin_offset + int(types[obj_idx])
        if [obj_node, bin_node] not in list(zip(src, dst)):
            src.append(obj_node)
            dst.append(bin_node)

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

    return node_features, edge_index, mask


# ═══════════════════════════════════════════════════════
# ALTERNATIVE MODEL ARCHITECTURES (A2)
# ═══════════════════════════════════════════════════════

class GCNPolicy(nn.Module):
    """GCN variant — fixed aggregation weights instead of attention."""
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(input_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, obj_mask):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        n_obj = obj_mask.size(0)
        h_obj = h[1:1 + n_obj]
        logits = self.head(h_obj).squeeze(-1)
        logits = logits.masked_fill(~obj_mask, MASK_FILL_VALUE)
        return logits


class MLPPolicy(nn.Module):
    """MLP baseline — no graph structure, just node features."""
    def __init__(self, input_dim, hidden_dim, n_objects, dropout):
        super().__init__()
        # Concatenate robot + all object + all bin features
        total_input = input_dim * (1 + n_objects + NUM_BINS)
        self.net = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_objects),
        )
        self.n_objects = n_objects

    def forward(self, x, edge_index, obj_mask):
        flat = x.flatten().unsqueeze(0)
        logits = self.net(flat).squeeze(0)
        logits = logits.masked_fill(~obj_mask, MASK_FILL_VALUE)
        return logits


# ═══════════════════════════════════════════════════════
# TRAINING LOOP (parameterised for ablations)
# ═══════════════════════════════════════════════════════

def train_ablation(
    name,
    train_data,
    val_data,
    device,
    model=None,
    graph_fn=None,
    stages=None,
    supervision="mixed",
    epochs_per_stage=DEFAULT_EPOCHS_PER_STAGE,
    epochs_final=DEFAULT_EPOCHS_FINAL,
):
    """Train one ablation variant and return eval metrics."""
    if graph_fn is None:
        graph_fn = _build_step_graph
    if model is None:
        model = GNNPolicy(FEATURE_DIM, DEFAULT_HIDDEN_DIM, DEFAULT_ATTENTION_HEADS, DEFAULT_DROPOUT).to(device)

    object_count = train_data[0]["objects"].size(0)
    if stages is None:
        stages = [p for p in PREFIX_STAGES if p <= object_count] or [object_count]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    stage_epochs = {p: (epochs_final if p == stages[-1] else epochs_per_stage) for p in stages}
    total_batches = sum(stage_epochs[p] * len(train_data) for p in stages)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_batches, 1))

    best_gap = -math.inf
    best_state = None
    global_step = 0
    variant_start = time.time()

    total_epochs = sum(stage_epochs[p] for p in stages)
    completed_epochs = 0

    print(f"\n{'='*70}")
    print(f"[{time.strftime('%H:%M:%S')}] --- {name} ---")
    print(f"  Stages: {stages}, Supervision: {supervision}")
    print(f"  Total epochs: {total_epochs}, Train samples: {len(train_data)}")

    for stage_idx, prefix in enumerate(stages):
        # Filter training set based on supervision mode
        if supervision == "ilp_only":
            stage_set = [s for s in train_data if str(prefix) in s["ilp_prefixes"]]
        elif supervision == "greedy_only":
            stage_set = [s for s in train_data if len(s.get("greedy_sequence", [])) >= prefix]
        else:  # mixed (default)
            stage_set = [s for s in train_data if str(prefix) in s["ilp_prefixes"]]
            if not stage_set:
                stage_set = [s for s in train_data
                             if "greedy_sequence" in s and len(s["greedy_sequence"]) >= prefix]

        if not stage_set:
            print(f"  [{time.strftime('%H:%M:%S')}] Stage {prefix}: no data, skipping")
            continue

        n_epochs = stage_epochs[prefix]
        print(f"  [{time.strftime('%H:%M:%S')}] Stage {prefix} ({stage_idx+1}/{len(stages)}): "
              f"{len(stage_set)} scenarios, {n_epochs} epochs")

        for epoch in range(n_epochs):
            epoch_start = time.time()
            rng = np.random.default_rng(SEED + epoch + prefix)
            indices = rng.permutation(len(stage_set))
            epoch_losses = []
            accumulated = 0

            for idx in indices:
                scenario = stage_set[idx]

                # Compute loss with the specified graph function
                loss = _compute_loss_with_graph(model, scenario, prefix, device, graph_fn, supervision)
                if loss is None:
                    continue

                loss = loss / DEFAULT_ACCUMULATE_GRAD
                loss.backward()
                accumulated += 1

                if accumulated % DEFAULT_ACCUMULATE_GRAD == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_losses.append(loss.item() * DEFAULT_ACCUMULATE_GRAD)

            if accumulated % DEFAULT_ACCUMULATE_GRAD != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Evaluate
            metrics = _evaluate_with_graph(model, val_data, device, stages[-1], graph_fn)
            gap = metrics["mean_gap_vs_greedy"]
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

            if gap > best_gap:
                best_gap = gap
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            completed_epochs += 1
            epoch_elapsed = time.time() - epoch_start
            total_elapsed = time.time() - variant_start
            eta_s = (total_elapsed / completed_epochs) * (total_epochs - completed_epochs)
            eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s" if eta_s > 60 else f"{eta_s:.0f}s"

            marker = " *BEST*" if gap >= best_gap else ""
            print(f"    [{time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{n_epochs} "
                  f"({completed_epochs}/{total_epochs} total) "
                  f"loss={avg_loss:.4f} gap={gap:+.2f}% win={metrics['win_rate']:.1f}% "
                  f"[{epoch_elapsed:.1f}s, ETA {eta_str}]{marker}")

    # Load best and final eval
    if best_state:
        model.load_state_dict(best_state)
    final_metrics = _evaluate_with_graph(model, val_data, device, stages[-1], graph_fn)
    variant_elapsed = time.time() - variant_start
    elapsed_str = f"{int(variant_elapsed//60)}m{int(variant_elapsed%60):02d}s"
    print(f"  [{time.strftime('%H:%M:%S')}] DONE in {elapsed_str}: "
          f"gap={final_metrics['mean_gap_vs_greedy']:+.2f}% "
          f"win={final_metrics['win_rate']:.1f}%")
    return final_metrics


def _compute_loss_with_graph(model, scenario, prefix_size, device, graph_fn, supervision):
    """Like compute_scenario_loss but uses a custom graph builder and supervision mode."""
    prefix_key = str(prefix_size)
    order = None

    if supervision == "greedy_only":
        greedy = scenario.get("greedy_sequence", [])
        if len(greedy) >= prefix_size:
            order = greedy[:prefix_size]
    elif supervision == "ilp_only":
        if prefix_key in scenario["ilp_prefixes"]:
            order = scenario["ilp_prefixes"][prefix_key]
    else:  # mixed
        if prefix_key in scenario["ilp_prefixes"]:
            order = scenario["ilp_prefixes"][prefix_key]
        elif "greedy_sequence" in scenario:
            greedy = scenario["greedy_sequence"]
            if len(greedy) >= prefix_size:
                order = greedy[:prefix_size]

    if not order:
        return None

    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    mask = torch.ones(objects.size(0), device=device, dtype=torch.bool)
    robot_world = scenario["start"]

    losses = []
    for target in order:
        nf, ei, om = graph_fn(objects, bins, types, mask, robot_world, device)
        logits = model(nf, ei, om)
        target_tensor = torch.tensor([target], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
        losses.append(loss)
        robot_world = bins[types[target]]
        mask[target] = False

    if not losses:
        return None
    return torch.stack(losses).mean()


def _rollout_with_graph(model, scenario, device, graph_fn):
    """Like rollout_model but uses a custom graph builder."""
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    mask = torch.ones(objects.size(0), device=device, dtype=torch.bool)
    robot_world = scenario["start"]
    total_cost = 0.0

    while mask.any():
        nf, ei, om = graph_fn(objects, bins, types, mask, robot_world, device)
        logits = model(nf, ei, om)
        action = int(torch.argmax(logits).item())
        if not mask[action]:
            break
        obj_pos = objects[action]
        bin_pos = bins[types[action]]
        total_cost += torch.norm(robot_world - obj_pos).item()
        total_cost += torch.norm(obj_pos - bin_pos).item()
        robot_world = bin_pos
        mask[action] = False

    return total_cost


def _evaluate_with_graph(model, scenarios, device, final_prefix, graph_fn):
    """Like evaluate() but uses a custom graph builder."""
    model.eval()
    costs, gaps = [], []
    win_count = 0
    with torch.no_grad():
        for scenario in scenarios:
            cost = _rollout_with_graph(model, scenario, device, graph_fn)
            greedy_cost = scenario["greedy_cost"]
            costs.append(cost)
            gap = (greedy_cost - cost) / greedy_cost * 100.0 if greedy_cost > 0 else 0.0
            gaps.append(gap)
            if cost < greedy_cost:
                win_count += 1

    model.train()
    return {
        "mean_cost": float(np.mean(costs)) if costs else 0.0,
        "mean_gap_vs_greedy": float(np.mean(gaps)) if gaps else 0.0,
        "gap_std": float(np.std(gaps)) if gaps else 0.0,
        "win_rate": (win_count / len(scenarios) * 100.0) if scenarios else 0.0,
    }


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {DATA_PATH}")

    scenarios = load_scenarios(DATA_PATH)
    if MAX_SCENARIOS and len(scenarios) > MAX_SCENARIOS:
        scenarios = scenarios[:MAX_SCENARIOS]
    train_raw, val_raw = split_dataset(scenarios, VAL_SPLIT, SEED)
    train_data = [prepare_scenario(s, device) for s in train_raw]
    val_data = [prepare_scenario(s, device) for s in val_raw]
    object_count = train_data[0]["objects"].size(0)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Objects: {object_count}")
    print(f"Started at: {time.strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    results = {}
    run_start = time.time()

    # ── A1: Graph Structure (3 variants) ──
    print(f"\n[{time.strftime('%H:%M:%S')}] ▶ ABLATION GROUP A1: Graph Structure (variants 1-3 of 9)")
    results["A1_cyclic_star"] = train_ablation(
        "[1/9] A1: Cyclic Star (ours)", train_data, val_data, device,
        graph_fn=_build_step_graph,
    )

    results["A1_fully_connected"] = train_ablation(
        "[2/9] A1: Fully Connected", train_data, val_data, device,
        graph_fn=_build_fc_graph,
    )

    results["A1_knn"] = train_ablation(
        "[3/9] A1: k-NN (k=5)", train_data, val_data, device,
        graph_fn=_build_knn_graph,
    )

    # ── A2: Encoder Architecture (2 new + reuse GAT) ──
    print(f"\n[{time.strftime('%H:%M:%S')}] ▶ ABLATION GROUP A2: Encoder Architecture (variants 4-5 of 9)")
    results["A2_gat"] = results["A1_cyclic_star"]

    gcn_model = GCNPolicy(FEATURE_DIM, DEFAULT_HIDDEN_DIM, DEFAULT_DROPOUT).to(device)
    results["A2_gcn"] = train_ablation(
        "[4/9] A2: GCN", train_data, val_data, device,
        model=gcn_model, graph_fn=_build_step_graph,
    )

    mlp_model = MLPPolicy(FEATURE_DIM, DEFAULT_HIDDEN_DIM, object_count, DEFAULT_DROPOUT).to(device)
    results["A2_mlp"] = train_ablation(
        "[5/9] A2: MLP (no graph)", train_data, val_data, device,
        model=mlp_model, graph_fn=_build_step_graph,
    )

    # ── A3: Curriculum (1 new + reuse baseline) ──
    print(f"\n[{time.strftime('%H:%M:%S')}] ▶ ABLATION GROUP A3: Curriculum (variant 6 of 9)")
    results["A3_curriculum"] = results["A1_cyclic_star"]

    results["A3_no_curriculum"] = train_ablation(
        "[6/9] A3: No Curriculum", train_data, val_data, device,
        stages=[object_count],
        epochs_final=DEFAULT_EPOCHS_PER_STAGE * len([p for p in PREFIX_STAGES if p <= object_count]) + DEFAULT_EPOCHS_FINAL,
    )

    # ── A4: Supervision Source (2 new + reuse baseline) ──
    print(f"\n[{time.strftime('%H:%M:%S')}] ▶ ABLATION GROUP A4: Supervision Source (variants 7-8 of 9)")
    results["A4_mixed"] = results["A1_cyclic_star"]

    results["A4_ilp_only"] = train_ablation(
        "[7/9] A4: ILP-only", train_data, val_data, device,
        supervision="ilp_only",
    )

    results["A4_greedy_only"] = train_ablation(
        "[8/9] A4: Greedy-only", train_data, val_data, device,
        supervision="greedy_only",
    )

    total_time = time.time() - run_start
    print(f"\n[{time.strftime('%H:%M:%S')}] All variants done in "
          f"{int(total_time//60)}m{int(total_time%60):02d}s")

    # ── Save results ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ablation_results_40obj.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Ablation':<30} {'Gap vs Greedy':>15} {'Win Rate':>10}")
    print(f"{'='*70}")
    for key, m in results.items():
        print(f"{key:<30} {m['mean_gap_vs_greedy']:>14.2f}% {m['win_rate']:>9.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
