"""
Batched-rollout version of `gnn_train.py`.

The original training loop ran one scenario per gradient step (T sequential
GAT forwards per scenario), saturating GPU at ~17% util. Here we expand every
scenario into T (mask, robot_pos, target) step examples, then batch K of them
into a single PyG `Batch` so each gradient step is one GPU forward over K
graphs. Same model, same data, same curriculum, same checkpoint format.

Run from src/:
  /home/pan-navigator/binning_venv/bin/python gnn_train_batched.py \\
    --dataset ../data/dataset_40_objects.json \\
    --model-out ../models/gnn_thin_h64_n40_curr_to_40.pt \\
    --hidden-dim 64 --heads 4 \\
    --epochs-per-stage 3 --epochs-final 10 \\
    --batch-size 64
"""
import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from gnn_train import (
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_VAL_SPLIT,
    DEFAULT_EPOCHS_PER_STAGE,
    DEFAULT_EPOCHS_FINAL,
    FEATURE_DIM,
    NUM_BINS,
    WORKSPACE_SIZE,
    MASK_FILL_VALUE,
    MAX_GRAD_NORM,
    PREFIX_STAGES,
    GNNPolicy,
    load_scenarios,
    split_dataset,
    prepare_scenario,
    evaluate,
    get_model_name,
)


def _build_step_data(
    objects: torch.Tensor,
    bins: torch.Tensor,
    types: torch.Tensor,
    mask: torch.Tensor,          # bool[n_objects]
    robot_world: torch.Tensor,   # [2]
    target_idx: int,
) -> Data:
    """Construct a PyG Data object for ONE step. CPU tensors; Batch.to(device) later."""
    n_objects = objects.size(0)
    bin_offset = 1 + n_objects
    robot_norm = robot_world / WORKSPACE_SIZE
    obj_norm = objects / WORKSPACE_SIZE
    bin_norm = bins / WORKSPACE_SIZE
    type_one_hot = F.one_hot(types, num_classes=NUM_BINS).float()

    robot_feat = torch.zeros((1, FEATURE_DIM))
    robot_feat[0, 0:2] = robot_norm

    object_feat = torch.zeros((n_objects, FEATURE_DIM))
    object_feat[:, 0:2] = obj_norm
    object_feat[:, 2:4] = bin_norm[types]
    object_feat[:, 4 : 4 + NUM_BINS] = type_one_hot
    object_feat[:, 4 + NUM_BINS] = mask.float()

    bin_feat = torch.zeros((NUM_BINS, FEATURE_DIM))
    bin_feat[:, 0:2] = bin_norm
    bin_feat[:, 2:4] = bin_norm
    bin_feat[:, 4 : 4 + NUM_BINS] = torch.eye(NUM_BINS)

    x = torch.cat([robot_feat, object_feat, bin_feat], dim=0)

    src, dst = [], []
    active = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    for obj_idx in active:
        obj_node = 1 + obj_idx
        bin_node = bin_offset + int(types[obj_idx])
        src.extend([0, obj_node, bin_node])
        dst.extend([obj_node, bin_node, 0])
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.obj_mask = mask.unsqueeze(0)        # [1, n_objects]
    data.n_objects = torch.tensor([n_objects])
    data.target = torch.tensor([target_idx], dtype=torch.long)
    return data


def enumerate_step_examples(scenarios: List[Dict], prefix_size: int) -> List[Data]:
    """For each scenario with an ILP order of length >= prefix_size, expand into
    `prefix_size` step Data objects (one per action). All CPU tensors."""
    examples: List[Data] = []
    pkey = str(prefix_size)
    for s in scenarios:
        order = None
        if pkey in s["ilp_prefixes"]:
            order = s["ilp_prefixes"][pkey]
        elif "greedy_sequence" in s and len(s["greedy_sequence"]) >= prefix_size:
            order = s["greedy_sequence"][:prefix_size]
        if not order:
            continue
        order = order[:prefix_size]

        objects_cpu = s["objects"].detach().cpu()
        bins_cpu = s["bins"].detach().cpu()
        types_cpu = s["types"].detach().cpu()
        start_cpu = s["start"].detach().cpu()

        n_objects = objects_cpu.size(0)
        mask = torch.ones(n_objects, dtype=torch.bool)
        robot = start_cpu.clone()
        for target in order:
            data = _build_step_data(objects_cpu, bins_cpu, types_cpu, mask, robot, int(target))
            examples.append(data)
            # advance state
            robot = bins_cpu[types_cpu[target]].clone()
            mask = mask.clone()
            mask[target] = False
    return examples


def forward_batch(model: nn.Module, batch: Batch, device: torch.device) -> torch.Tensor:
    """Run GNNPolicy on a PyG Batch and compute summed cross-entropy across K graphs.
    Returns scalar loss."""
    batch = batch.to(device, non_blocking=True)

    # Run convs on the flattened node feature matrix.
    h = batch.x
    for conv in model.convs:
        h = conv(h, batch.edge_index)
        h = F.relu(h)
        h = model.dropout(h)
    # Compute per-node logits, then for each graph extract its object slice.
    logits_all = model.head(h).squeeze(-1)   # [total_nodes]

    # Per-graph slicing. PyG `Batch` provides `batch.ptr` (cumulative node counts).
    # Object indices in each graph: nodes [1, 1 + n_objects).
    ptr = batch.ptr.tolist()
    n_objs = batch.n_objects.tolist()
    targets = batch.target.to(device)

    total_loss = 0.0
    K = len(n_objs)
    for k in range(K):
        start = ptr[k]
        n = n_objs[k]
        obj_logits = logits_all[start + 1 : start + 1 + n]
        mask_k = batch.obj_mask[k].to(device)
        obj_logits = obj_logits.masked_fill(~mask_k, MASK_FILL_VALUE)
        total_loss = total_loss + F.cross_entropy(obj_logits.unsqueeze(0), targets[k:k+1])
    return total_loss / K


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model-out", type=str, required=True)
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--heads", type=int, default=DEFAULT_ATTENTION_HEADS)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--epochs-per-stage", type=int, default=DEFAULT_EPOCHS_PER_STAGE)
    p.add_argument("--epochs-final", type=int, default=DEFAULT_EPOCHS_FINAL)
    p.add_argument("--batch-size", type=int, default=64, help="K scenarios per gradient step")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}  batch_size={args.batch_size}")

    scenarios = load_scenarios(Path(args.dataset))
    train_raw, val_raw = split_dataset(scenarios, args.val_split, args.seed)
    print(f"train={len(train_raw)} val={len(val_raw)}")

    # Prepare once on CPU (move per-batch to GPU)
    train_data = [prepare_scenario(s, torch.device("cpu")) for s in train_raw]
    val_data = [prepare_scenario(s, device) for s in val_raw]

    object_count = max(s["objects"].size(0) for s in train_data)
    stages = [p for p in PREFIX_STAGES if p <= object_count] or [object_count]
    print(f"Curriculum stages: {stages}")
    stage_epochs = {p: (args.epochs_final if p == stages[-1] else args.epochs_per_stage) for p in stages}

    model = GNNPolicy(FEATURE_DIM, args.hidden_dim, args.heads, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Optimizer + cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Pre-enumerate step examples per stage (once)
    stage_examples = {}
    for prefix in stages:
        t0 = time.time()
        ex = enumerate_step_examples(train_data, prefix)
        stage_examples[prefix] = ex
        print(f"Stage {prefix}: {len(ex):,} step examples ({time.time()-t0:.1f}s)")

    total_iters = sum(stage_epochs[p] * (len(stage_examples[p]) // args.batch_size + 1) for p in stages)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-6)
    print(f"Total optimizer steps: ~{total_iters:,}")

    best_gap = -1e9
    best_state = None
    safe_dir = Path("/tmp/binning_checkpoints")
    safe_dir.mkdir(parents=True, exist_ok=True)
    model_out = Path(args.model_out)
    safe_path = safe_dir / f"{model_out.stem}_best.pt"

    global_epoch = 0
    for prefix in stages:
        examples = stage_examples[prefix]
        if not examples:
            print(f"[skip] stage {prefix} has no examples")
            continue
        n_batches = (len(examples) + args.batch_size - 1) // args.batch_size

        pbar = tqdm(range(stage_epochs[prefix]), desc=f"Stage {prefix}", leave=True)
        for epoch in pbar:
            rng = np.random.default_rng(args.seed + global_epoch)
            idx = rng.permutation(len(examples))

            model.train()
            t_epoch = time.time()
            epoch_losses = []
            for b in range(n_batches):
                bi = idx[b * args.batch_size : (b + 1) * args.batch_size]
                if len(bi) == 0:
                    continue
                batch = Batch.from_data_list([examples[i] for i in bi])
                loss = forward_batch(model, batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                epoch_losses.append(loss.item())
            epoch_dt = time.time() - t_epoch

            # Evaluate on val (uses sequential rollout from gnn_train.evaluate)
            metrics = evaluate(model, val_data, device, final_prefix=stages[-1])
            gap = metrics["mean_gap_vs_greedy"]
            is_best = gap > best_gap
            if is_best:
                best_gap = gap
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    'state_dict': best_state,
                    'hidden_dim': args.hidden_dim,
                    'heads': args.heads,
                    'dropout': args.dropout,
                    'object_count': object_count,
                    'model_type': 'gnn',
                    'best_gap_at_save': best_gap,
                    'stage': prefix,
                }, safe_path)
            pbar.set_postfix(
                loss=f"{np.mean(epoch_losses):.4f}",
                gap_vs_greedy=f"{gap:.2f}",
                best_gap=f"{best_gap:.2f}",
                epoch_s=f"{epoch_dt:.1f}",
            )
            global_epoch += 1

    # Restore best, save permanently
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = {
        'state_dict': model.state_dict(),
        'hidden_dim': args.hidden_dim,
        'heads': args.heads,
        'dropout': args.dropout,
        'object_count': object_count,
        'model_type': 'gnn',
    }
    final_safe = safe_dir / f"{model_out.stem}_final.pt"
    torch.save(checkpoint, final_safe)
    print(f"Saved (robust): {final_safe}")

    descriptive_name = get_model_name(object_count, "gnn", "best")
    for tries in range(3):
        try:
            model_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final_safe, model_out)
            shutil.copyfile(final_safe, model_out.parent / descriptive_name)
            print(f"Copied to: {model_out} and {model_out.parent / descriptive_name}")
            break
        except OSError as e:
            print(f"  [warn] copy attempt {tries+1}/3 failed: {e}")
            time.sleep(5)
    else:
        print(f"  [warn] mount copy failed; final at {final_safe}")

    # Final metrics
    final_metrics = evaluate(model, val_data, device, final_prefix=stages[-1])
    print(f"\nFinal validation: {final_metrics}")


if __name__ == "__main__":
    main()
