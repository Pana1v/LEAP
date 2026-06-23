import argparse
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from tqdm import tqdm

# Training logger for comprehensive logging and visualization
from training_logger import TrainingLogger, extract_dataset_info
# Paper metrics logger for publication-ready outputs
from paper_metrics_logger import PaperMetricsLogger, get_model_name

# Core configuration (avoid magic numbers)
WORKSPACE_SIZE = 100.0
NUM_BINS = 4
PREFIX_STAGES = [5, 10, 20, 40]
DEFAULT_EPOCHS_PER_STAGE = 3  # Increased from 2 for better convergence
DEFAULT_EPOCHS_FINAL = 6  # Increased from 4
DEFAULT_HIDDEN_DIM = 128
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 1e-3  # Increased from 3e-4 - critical for learning
DEFAULT_WEIGHT_DECAY = 0.01  # Increased from 1e-4 for better generalization
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_ACCUMULATE_GRAD = 8  # Gradient accumulation to reduce noise, prevent greedy collapse
MAX_GRAD_NORM = 1.0
MASK_FILL_VALUE = -1e4
FEATURE_DIM = 4 + NUM_BINS + 1  # pos(obj), pos(bin), type one-hot, mask slot


def load_scenarios(path: Path, object_count: int = None) -> List[Dict]:
    with open(path, "r") as f:
        scenarios = json.load(f)
    if object_count is not None:
        scenarios = [s for s in scenarios if len(s.get("objects", [])) == object_count]
    return scenarios


def split_dataset(scenarios: List[Dict], val_split: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(scenarios))
    rng.shuffle(indices)
    val_size = int(len(indices) * val_split)
    val_idx = set(indices[:val_size].tolist())
    train, val = [], []
    for i, s in enumerate(scenarios):
        (val if i in val_idx else train).append(s)
    return train, val


def _build_step_graph(
    objects: torch.Tensor,
    bins: torch.Tensor,
    types: torch.Tensor,
    mask: torch.Tensor,
    robot_world: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a star-style graph for the current step:
    - Nodes: robot, objects, bins.
    - Edges: robot->object (pick), object->bin (place), bin->robot (feedback).
    Returns node_features, edge_index, object_mask.
    """
    n_objects = objects.size(0)
    bin_offset = 1 + n_objects
    robot_norm = robot_world / WORKSPACE_SIZE
    obj_norm = objects / WORKSPACE_SIZE
    bin_norm = bins / WORKSPACE_SIZE
    type_one_hot = F.one_hot(types, num_classes=NUM_BINS).float()

    # Robot feature padded to FEATURE_DIM
    robot_feat = torch.zeros((1, FEATURE_DIM), device=device)
    robot_feat[0, 0:2] = robot_norm

    # Object features
    object_feat = torch.zeros((n_objects, FEATURE_DIM), device=device)
    object_feat[:, 0:2] = obj_norm
    object_feat[:, 2:4] = bin_norm[types]
    object_feat[:, 4 : 4 + NUM_BINS] = type_one_hot
    object_feat[:, 4 + NUM_BINS] = mask.float()

    # Bin features
    bin_feat = torch.zeros((NUM_BINS, FEATURE_DIM), device=device)
    bin_feat[:, 0:2] = bin_norm
    bin_feat[:, 2:4] = bin_norm
    bin_feat[:, 4 : 4 + NUM_BINS] = torch.eye(NUM_BINS, device=device)

    node_features = torch.cat([robot_feat, object_feat, bin_feat], dim=0)

    src, dst = [], []
    for obj_idx in torch.nonzero(mask, as_tuple=False).flatten():
        obj_node = 1 + obj_idx.item()
        bin_node = bin_offset + int(types[obj_idx])
        src.extend([0, obj_node, bin_node])
        dst.extend([obj_node, bin_node, 0])

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

    return node_features, edge_index, mask


class GNNPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                GATConv(input_dim, hidden_dim, heads=heads, concat=False, dropout=dropout),
                GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, obj_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns logits for object nodes only; obj_mask length equals number of objects.
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        n_obj = obj_mask.size(0)
        h_obj = h[1 : 1 + n_obj]
        logits = self.head(h_obj).squeeze(-1)
        logits = logits.masked_fill(~obj_mask, MASK_FILL_VALUE)
        return logits


def prepare_scenario(scenario: Dict, device: torch.device) -> Dict:
    objects = torch.tensor(scenario["objects"], dtype=torch.float32, device=device)
    bins = torch.tensor(scenario["bins"], dtype=torch.float32, device=device)
    types = torch.tensor(scenario["types"], dtype=torch.long, device=device)
    start = torch.tensor(scenario["start"], dtype=torch.float32, device=device)
    ilp_prefixes = scenario.get("ilp_prefixes", {})
    ilp_costs = scenario.get("ilp_costs", {})
    return {
        "objects": objects,
        "bins": bins,
        "types": types,
        "start": start,
        "ilp_prefixes": ilp_prefixes,
        "ilp_costs": ilp_costs,
        "greedy_cost": scenario.get("greedy_cost", 0.0),
        "greedy_sequence": scenario.get("greedy_sequence", []),
    }


def compute_scenario_loss(
    model: nn.Module,
    scenario: Dict,
    prefix_size: int,
    device: torch.device,
) -> torch.Tensor:
    prefix_key = str(prefix_size)
    order = None
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
        node_features, edge_index, obj_mask = _build_step_graph(
            objects=objects,
            bins=bins,
            types=types,
            mask=mask,
            robot_world=robot_world,
            device=device,
        )
        logits = model(node_features, edge_index, obj_mask)
        target_tensor = torch.tensor([target], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
        losses.append(loss)

        obj_pos = objects[target]
        bin_pos = bins[types[target]]
        robot_world = bin_pos
        mask[target] = False

    if not losses:
        return None
    return torch.stack(losses).mean()


def rollout_model(model: nn.Module, scenario: Dict, device: torch.device) -> float:
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]

    mask = torch.ones(objects.size(0), device=device, dtype=torch.bool)
    robot_world = scenario["start"]
    total_cost = 0.0
    steps = 0

    while mask.any():
        node_features, edge_index, obj_mask = _build_step_graph(
            objects=objects,
            bins=bins,
            types=types,
            mask=mask,
            robot_world=robot_world,
            device=device,
        )
        logits = model(node_features, edge_index, obj_mask)
        action = int(torch.argmax(logits).item())
        if not mask[action]:
            break

        obj_pos = objects[action]
        bin_pos = bins[types[action]]
        total_cost += torch.norm(robot_world - obj_pos).item()
        total_cost += torch.norm(obj_pos - bin_pos).item()

        robot_world = bin_pos
        mask[action] = False
        steps += 1
        if steps > objects.size(0) + 1:
            break

    return total_cost


def _random_cost(scenario: Dict) -> float:
    objects = scenario["objects"].cpu()
    bins = scenario["bins"].cpu()
    types = scenario["types"].cpu()
    start = scenario["start"].cpu()
    order = torch.randperm(objects.size(0))
    cur = start
    cost = 0.0
    for idx in order:
        obj = objects[idx]
        bin_pos = bins[types[idx]]
        cost += torch.norm(cur - obj).item()
        cost += torch.norm(obj - bin_pos).item()
        cur = bin_pos
    return cost


def evaluate(model: nn.Module, scenarios: List[Dict], device: torch.device, final_prefix: int) -> Dict:
    model.eval()
    costs, gaps, ilp_gaps, random_costs = [], [], [], []
    win_count = 0
    with torch.no_grad():
        for scenario in scenarios:
            cost = rollout_model(model, scenario, device)
            greedy_cost = scenario["greedy_cost"]
            costs.append(cost)
            gap = (greedy_cost - cost) / greedy_cost * 100.0 if greedy_cost > 0 else 0.0
            gaps.append(gap)
            if cost < greedy_cost:
                win_count += 1
            ilp_cost = scenario["ilp_costs"].get(str(final_prefix))
            if ilp_cost is not None:
                ilp_gaps.append((ilp_cost - cost) / ilp_cost * 100.0 if ilp_cost > 0 else 0.0)
            random_costs.append(_random_cost(scenario))

    win_rate = (win_count / len(scenarios) * 100.0) if scenarios else 0.0
    gap_std = float(np.std(gaps)) if gaps else 0.0

    result = {
        "mean_cost": float(np.mean(costs)) if costs else 0.0,
        "mean_gap_vs_greedy": float(np.mean(gaps)) if gaps else 0.0,
        "gap_std": gap_std,
        "win_rate": win_rate,
        "mean_gap_vs_ilp": float(np.mean(ilp_gaps)) if ilp_gaps else None,
        "mean_cost_random": float(np.mean(random_costs)) if random_costs else 0.0,
    }
    model.train()
    return result


def train_model(
    dataset_path: Path,
    model_out: Path,
    epochs_per_stage: int,
    epochs_final: int,
    hidden_dim: int,
    heads: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    val_split: float,
    seed: int,
    device: torch.device,
    accumulate_grad: int = DEFAULT_ACCUMULATE_GRAD,
    enable_logging: bool = True,
    log_dir: Optional[str] = None,
):
    """
    Train the GNN policy model.

    Args:
        dataset_path: Path to the dataset JSON file
        model_out: Path to save the trained model
        epochs_per_stage: Number of epochs per curriculum stage
        epochs_final: Number of epochs for the final stage
        hidden_dim: Hidden dimension of the model
        heads: Number of attention heads
        dropout: Dropout rate
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        val_split: Validation split fraction
        seed: Random seed
        device: Device to train on
        accumulate_grad: Gradient accumulation steps (default: 8)
        enable_logging: Whether to enable comprehensive logging
        log_dir: Base directory for logs (default: 'logs')
    """
    scenarios = load_scenarios(dataset_path)
    if not scenarios:
        raise ValueError(f"No scenarios found in {dataset_path}")
    train_raw, val_raw = split_dataset(scenarios, val_split, seed)

    train_data = [prepare_scenario(s, device) for s in train_raw]
    val_data = [prepare_scenario(s, device) for s in val_raw]

    object_count = train_data[0]["objects"].size(0)
    stages = [p for p in PREFIX_STAGES if p <= object_count] or [object_count]

    # Initialize training logger
    logger = None
    paper_logger = None
    if enable_logging:
        dataset_name = extract_dataset_info(dataset_path)
        hyperparams = {
            "epochs_per_stage": epochs_per_stage,
            "epochs_final": epochs_final,
            "hidden_dim": hidden_dim,
            "heads": heads,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "val_split": val_split,
            "accumulate_grad": accumulate_grad,
            "seed": seed,
            "device": str(device),
            "num_train_scenarios": len(train_data),
            "num_val_scenarios": len(val_data),
            "object_count": object_count,
            "stages": stages,
        }
        logger = TrainingLogger(
            dataset_name=dataset_name,
            model_type="gnn",
            base_log_dir=log_dir or "logs",
            hyperparams=hyperparams,
        )

        # Initialize paper metrics logger for publication-ready outputs
        paper_logger = PaperMetricsLogger(
            dataset_path=dataset_path,
            model_type="gnn",
            experiment_name=f"gnn_baseline_{object_count}obj",
            base_dir=log_dir or "experiments",
            hyperparams=hyperparams,
        )

    input_dim = FEATURE_DIM
    model = GNNPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    stage_epochs = {p: (epochs_final if p == stages[-1] else epochs_per_stage) for p in stages}

    def stage_batch_count(prefix: int) -> int:
        return sum(
            1
            for s in train_data
            if (str(prefix) in s["ilp_prefixes"]) or (len(s.get("greedy_sequence", [])) >= prefix)
        )

    total_batches = sum(stage_epochs[p] * stage_batch_count(p) for p in stages)
    total_batches = max(total_batches, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches)

    best_gap = -math.inf
    best_state = None
    global_step = 0
    global_epoch = 0
    current_stage = 0

    print(f"\n{'='*60}")
    print(f"Training GNN on {dataset_path.name}")
    print(f"Stages: {stages}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    for prefix in stages:
        stage_set = [s for s in train_data if str(prefix) in s["ilp_prefixes"]]
        if not stage_set:
            # allow fallback to greedy supervision if ILP prefix absent
            stage_set = [s for s in train_data if "greedy_sequence" in s and len(s["greedy_sequence"]) >= prefix]
        if not stage_set:
            continue

        # Log stage transition
        if logger and current_stage != prefix:
            if current_stage > 0:
                logger.log_stage_transition(current_stage, prefix, global_epoch)
            current_stage = prefix

        pbar = tqdm(range(stage_epochs[prefix]), desc=f"Stage {prefix}", leave=True)
        for epoch in pbar:
            rng = np.random.default_rng(seed + epoch + prefix)
            indices = rng.permutation(len(stage_set))

            epoch_losses = []
            accumulated = 0
            grad_norms = []

            for idx in indices:
                scenario = stage_set[idx]
                loss = compute_scenario_loss(model, scenario, prefix, device)
                if loss is None:
                    continue

                # Accumulate gradient over multiple scenarios
                loss = loss / accumulate_grad
                loss.backward()
                accumulated += 1

                if accumulated % accumulate_grad == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_losses.append(loss.item() * accumulate_grad)  # Store un-accumulated loss for reporting

            # Flush remaining accumulated gradients at end of epoch
            if accumulated % accumulate_grad != 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            global_epoch += 1

            # Evaluate once per epoch within stage
            metrics = evaluate(model, val_data, device, final_prefix=stages[-1])
            gap = metrics["mean_gap_vs_greedy"]

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Log epoch data
            is_best = gap > best_gap
            if logger:
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
                logger.log_epoch(
                    epoch=global_epoch,
                    stage=prefix,
                    train_loss=avg_loss,
                    val_metrics=metrics,
                    learning_rate=current_lr,
                    grad_norm=avg_grad_norm,
                )

            # Log to paper metrics logger (CSV + JSON)
            if paper_logger:
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
                paper_logger.log_epoch(
                    epoch=global_epoch,
                    stage=prefix,
                    train_loss=avg_loss,
                    val_metrics=metrics,
                    learning_rate=current_lr,
                    grad_norm=avg_grad_norm,
                    is_best=is_best,
                )

            if gap > best_gap:
                best_gap = gap
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                try:
                    _safe_dir = Path("/tmp/binning_checkpoints")
                    _safe_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'state_dict': best_state,
                        'hidden_dim': hidden_dim,
                        'heads': heads,
                        'dropout': dropout,
                        'object_count': object_count,
                        'model_type': 'gnn',
                        'best_gap_at_save': best_gap,
                        'stage': prefix,
                    }, _safe_dir / f"{model_out.stem}_best.pt")
                except Exception as _e:
                    print(f"  [warn] safe-checkpoint failed: {_e}")

            pbar.set_postfix(
                gap_vs_greedy=f"{metrics['mean_gap_vs_greedy']:.2f}",
                mean_cost=f"{metrics['mean_cost']:.1f}",
                best_gap=f"{best_gap:.2f}",
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model with descriptive naming.
    # Write to /tmp first so the mount can't lose the weights on a glitch,
    # then copy across (with retry).
    checkpoint_data = {
        'state_dict': model.state_dict(),
        'hidden_dim': hidden_dim,
        'heads': heads,
        'dropout': dropout,
        'object_count': object_count,
        'model_type': 'gnn',
    }
    _safe_dir = Path("/tmp/binning_checkpoints")
    _safe_dir.mkdir(parents=True, exist_ok=True)
    _safe_path = _safe_dir / f"{model_out.stem}_final.pt"
    torch.save(checkpoint_data, _safe_path)
    print(f"Model saved (robust): {_safe_path}")

    import shutil
    descriptive_name = get_model_name(object_count, "gnn", "best")
    for tries in range(3):
        try:
            model_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_safe_path, model_out)
            shutil.copyfile(_safe_path, model_out.parent / descriptive_name)
            print(f"Model copied to: {model_out} and {model_out.parent / descriptive_name}")
            break
        except OSError as _e:
            print(f"  [warn] copy attempt {tries+1}/3 failed: {_e}")
            import time as _time
            _time.sleep(5)
    else:
        print(f"  [warn] final mount copy failed; weights at {_safe_path}")

    final_metrics = evaluate(model, val_data, device, final_prefix=stages[-1])
    print("\nFinal validation:", final_metrics)

    # Benchmark inference time
    inference_stats = None
    if paper_logger:
        import time
        print("\nBenchmarking inference time...")
        paper_logger.log_model_info(model)

        def inference_fn():
            scenario = val_data[np.random.randint(len(val_data))]
            return rollout_model(model, scenario, device)

        inference_stats = paper_logger.benchmark_inference(
            model=model,
            inference_fn=inference_fn,
            num_samples=100,
            warmup_samples=10,
            device=device,
        )

        # Save model with paper logger
        paper_logger.save_model(
            model=model,
            optimizer=optimizer,
            suffix="best",
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            final_metrics=final_metrics,
        )

    # Log final evaluation and generate plots
    if logger:
        logger.log_final_evaluation(final_metrics, beam_search=False)
        logger.generate_plots()
        logger.save()

    # Finalize paper logger
    if paper_logger:
        paper_logger.log_final_evaluation(
            metrics=final_metrics,
            beam_search=False,
            inference_stats=inference_stats,
        )
        paper_logger.finalize()

    return model, final_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN to imitate ILP prefixes and beat greedy.")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "data/dataset_20_objects.json"), help="Path to 20-object dataset.")
    parser.add_argument("--model-out", type=str, default=str(PROJECT_ROOT / "models/gnn_ilp.pt"), help="Path to save trained model.")
    parser.add_argument("--epochs-per-stage", type=int, default=DEFAULT_EPOCHS_PER_STAGE, help="Epochs for 5/10/15 prefixes.")
    parser.add_argument("--epochs-final", type=int, default=DEFAULT_EPOCHS_FINAL, help="Epochs for 20-prefix stage.")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden dimension.")
    parser.add_argument("--heads", type=int, default=DEFAULT_ATTENTION_HEADS, help="Attention heads for GAT.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT, help="Validation split fraction.")
    parser.add_argument("--accumulate-grad", type=int, default=DEFAULT_ACCUMULATE_GRAD, help="Gradient accumulation steps (reduce noise).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu or cuda).")
    # Logging arguments
    parser.add_argument("--log-dir", type=str, default="logs", help="Base directory for training logs.")
    parser.add_argument("--no-log", action="store_true", help="Disable logging and plot generation.")
    return parser.parse_args()


def resolve_device(cli_device: str) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    train_model(
        dataset_path=Path(args.dataset),
        model_out=Path(args.model_out),
        epochs_per_stage=args.epochs_per_stage,
        epochs_final=args.epochs_final,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        accumulate_grad=args.accumulate_grad,
        seed=args.seed,
        device=device,
        enable_logging=not args.no_log,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
