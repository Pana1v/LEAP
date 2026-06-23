"""
GNN Training GUI - Gradio dashboard for the paper's GAT model.
Uses model, graph construction, and training functions from gnn_train.py.
Run with: python3 gnn_gui.py
"""

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

from gnn_train import (
    GNNPolicy,
    _build_step_graph,
    load_scenarios as _load_scenarios,
    split_dataset,
    prepare_scenario as _prepare_scenario,
    compute_scenario_loss,
    rollout_model,
    evaluate,
    WORKSPACE_SIZE,
    NUM_BINS,
    FEATURE_DIM,
    PREFIX_STAGES,
    MAX_GRAD_NORM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_DROPOUT,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_VAL_SPLIT,
    DEFAULT_ACCUMULATE_GRAD,
    DEFAULT_EPOCHS_PER_STAGE,
    DEFAULT_EPOCHS_FINAL,
)
from training_logger import TrainingLogger, extract_dataset_info

MODEL_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "gnn_200obj_best.pt"
BIN_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
BIN_NAMES = ["Bin 0", "Bin 1", "Bin 2", "Bin 3"]


# ═══════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════

def load_scenarios(path):
    return _load_scenarios(Path(path))


def prepare_scenario(s, device):
    return _prepare_scenario(s, device)


# ═══════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════

def _infer_gnn_arch(sd: dict) -> Tuple[int, int, float]:
    """Infer hidden_dim and heads from a GNNPolicy state_dict's weight shapes."""
    if "convs.0.bias" in sd:
        hidden_dim = sd["convs.0.bias"].shape[0]
    elif "head.0.weight" in sd:
        hidden_dim = sd["head.0.weight"].shape[1]
    else:
        hidden_dim = DEFAULT_HIDDEN_DIM
    if "convs.0.att_src" in sd:
        heads = sd["convs.0.att_src"].shape[1]
    else:
        heads = DEFAULT_ATTENTION_HEADS
    return hidden_dim, heads, DEFAULT_DROPOUT


_GAT_KEYS = {"convs.0.lin.weight", "convs.0.att_src", "head.0.weight"}


def _is_gat_checkpoint(sd: dict) -> bool:
    return bool(_GAT_KEYS & sd.keys())


def load_model(device, model_path=None):
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
        hd = int(blob.get("hidden_dim", 0))
        heads = int(blob.get("heads", 0))
        dp = float(blob.get("dropout", DEFAULT_DROPOUT))
        if not hd or not heads:
            hd, heads, dp = _infer_gnn_arch(sd)
    else:
        sd = blob
        hd, heads, dp = _infer_gnn_arch(sd)
    if not _is_gat_checkpoint(sd):
        raise RuntimeError(
            f"{path.name} is not a GNNPolicy (GAT) checkpoint — "
            f"it was likely saved by an older TransformerPointerNet. "
            f"Train a new model with the current architecture."
        )
    model = GNNPolicy(FEATURE_DIM, hd, heads, dp).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════

def greedy_rollout(scenario):
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=objects.device)
    robot = scenario["start"].clone()
    cost = 0.0
    for _ in range(n):
        valid = torch.nonzero(mask, as_tuple=False).flatten()
        if valid.numel() == 0:
            break
        pick_d = torch.norm(objects[valid] - robot.unsqueeze(0), dim=1)
        place_d = torch.norm(objects[valid] - bins[types[valid]], dim=1)
        best = valid[torch.argmin(pick_d + place_d)]
        cost += torch.norm(robot - objects[best]).item() + torch.norm(objects[best] - bins[types[best]]).item()
        robot = bins[types[best]]
        mask[best] = False
    return cost


def greedy_rollout_with_route(scenario):
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=objects.device)
    robot = scenario["start"].clone()
    cost = 0.0
    route = []
    for _ in range(n):
        valid = torch.nonzero(mask, as_tuple=False).flatten()
        if valid.numel() == 0:
            break
        pick_d = torch.norm(objects[valid] - robot.unsqueeze(0), dim=1)
        place_d = torch.norm(objects[valid] - bins[types[valid]], dim=1)
        best = valid[torch.argmin(pick_d + place_d)]
        sc = torch.norm(robot - objects[best]).item() + torch.norm(objects[best] - bins[types[best]]).item()
        cost += sc
        route.append(best.item())
        robot = bins[types[best]]
        mask[best] = False
    return cost, route


def gnn_rollout_with_route(model, scenario, device):
    model.eval()
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=device)
    robot = scenario["start"].clone()
    cost = 0.0
    route = []
    with torch.no_grad():
        for _ in range(n):
            nf, ei, om = _build_step_graph(objects, bins, types, mask, robot, device)
            logits = model(nf, ei, om)
            action = int(torch.argmax(logits).item())
            if not mask[action]:
                break
            sc = torch.norm(robot - objects[action]).item() + torch.norm(objects[action] - bins[types[action]]).item()
            cost += sc
            route.append(action)
            robot = bins[types[action]]
            mask[action] = False
    return cost, route


def rollout_timed(model, scenario, device):
    """Rollout with timing measurements for inference profiling."""
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    n = objects.size(0)
    mask = torch.ones(n, dtype=torch.bool, device=device)
    robot = scenario["start"].clone()
    cost = 0.0
    per_step_times = []
    total_forward = 0.0
    total_graph = 0.0
    sync = device.type == "cuda"

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            ts = time.perf_counter()

            tg0 = time.perf_counter()
            nf, ei, om = _build_step_graph(objects, bins, types, mask, robot, device)
            if sync:
                torch.cuda.synchronize()
            total_graph += time.perf_counter() - tg0

            tf0 = time.perf_counter()
            logits = model(nf, ei, om)
            if sync:
                torch.cuda.synchronize()
            total_forward += time.perf_counter() - tf0

            action = int(torch.argmax(logits).item())
            if not mask[action]:
                break
            cost += torch.norm(robot - objects[action]).item() + torch.norm(objects[action] - bins[types[action]]).item()
            robot = bins[types[action]]
            mask[action] = False
            per_step_times.append(time.perf_counter() - ts)

    return cost, {
        "total_time": time.perf_counter() - t0,
        "model_forward_time": total_forward,
        "graph_build_time": total_graph,
        "per_step_times": per_step_times,
    }


def random_cost(scenario):
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    perm = torch.randperm(objects.size(0), device=objects.device)
    robot = scenario["start"].clone()
    cost = 0.0
    for i in perm:
        cost += (torch.norm(robot - objects[i]) + torch.norm(objects[i] - bins[types[i]])).item()
        robot = bins[types[i]]
    return cost


def solve_ilp_route(scenario_raw, max_objects=60):
    if not ORTOOLS_AVAILABLE:
        return None, None, None
    start_time = time.time()
    objects = np.array(scenario_raw["objects"], dtype=np.float32)
    bins_arr = np.array(scenario_raw["bins"], dtype=np.float32)
    types_arr = np.array(scenario_raw["types"], dtype=np.int32)
    start = np.array(scenario_raw["start"], dtype=np.float32)
    n = len(objects)
    if n == 0:
        return 0.0, [], time.time() - start_time
    if n > max_objects:
        return None, None, time.time() - start_time

    pick_bin_cost = float(np.sum(np.linalg.norm(objects - bins_arr[types_arr], axis=1)))
    node_count = n + 1

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    costs = [[0.0] * node_count for _ in range(node_count)]
    for j in range(1, node_count):
        costs[0][j] = dist(start, objects[j - 1])
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i != j:
                costs[i][j] = dist(bins_arr[types_arr[i - 1]], objects[j - 1])

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return None, None, time.time() - start_time

    x = {}
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                x[i, j] = solver.BoolVar(f"x_{i}_{j}")

    for i in range(node_count):
        solver.Add(sum(x[i, j] for j in range(node_count) if j != i) == 1)
    for j in range(node_count):
        solver.Add(sum(x[i, j] for i in range(node_count) if i != j) == 1)

    u = [solver.NumVar(0, n, f"u_{i}") for i in range(node_count)]
    solver.Add(u[0] == 0)
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i != j:
                solver.Add(u[i] - u[j] + 1 <= n * (1 - x[i, j]))

    solver.Minimize(solver.Sum(costs[i][j] * x[i, j] for i in range(node_count) for j in range(node_count) if i != j))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return None, None, time.time() - start_time

    order_nodes = []
    current = 0
    visited = {0}
    while True:
        next_j = None
        for j in range(node_count):
            if j != current and x[current, j].solution_value() > 0.5:
                next_j = j
                break
        if next_j is None or next_j == 0:
            break
        order_nodes.append(next_j)
        if next_j in visited:
            break
        visited.add(next_j)
        current = next_j

    return solver.Objective().Value() + pick_bin_cost, [nd - 1 for nd in order_nodes], time.time() - start_time


# ═══════════════════════════════════════════════════════
# TRAINING STATE
# ═══════════════════════════════════════════════════════

@dataclass
class TrainingState:
    running: bool = False
    stop_requested: bool = False
    epoch: int = 0
    total_epochs: int = 0
    current_stage: int = 0
    stages: list = field(default_factory=list)
    stage_transitions: list = field(default_factory=list)
    train_losses: list = field(default_factory=list)
    val_costs: list = field(default_factory=list)
    greedy_mean: float = 0.0
    gaps: list = field(default_factory=list)
    best_gap: float = 0.0
    status: str = "Idle"
    batch_losses: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)
    grad_norms: list = field(default_factory=list)
    logs: list = field(default_factory=list)

    def reset(self):
        for f in self.__dataclass_fields__:
            v = self.__dataclass_fields__[f].default
            if v is not dataclass_sentinel:
                setattr(self, f, v)
            else:
                setattr(self, f, self.__dataclass_fields__[f].default_factory())


# Workaround: dataclass sentinel
dataclass_sentinel = getattr(TrainingState.__dataclass_fields__["running"], "default", None)

state = TrainingState()


def _reset_state():
    state.running = False
    state.stop_requested = False
    state.epoch = 0
    state.total_epochs = 0
    state.current_stage = 0
    state.stages = []
    state.stage_transitions = []
    state.train_losses = []
    state.val_costs = []
    state.greedy_mean = 0.0
    state.gaps = []
    state.best_gap = 0.0
    state.status = "Idle"
    state.batch_losses = []
    state.learning_rates = []
    state.grad_norms = []
    state.logs = []


# ═══════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════

def train_async(dataset_path, epochs_per_stage, epochs_final, train_limit,
                hidden_dim, heads, dropout, lr, weight_decay, val_split, seed,
                accumulate_grad=DEFAULT_ACCUMULATE_GRAD):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state.status = f"Loading data on {device}..."
        state.logs.append(f"Device: {device}")

        scenarios = load_scenarios(dataset_path)
        if train_limit and train_limit < len(scenarios):
            scenarios = scenarios[:int(train_limit)]

        train_raw, val_raw = split_dataset(scenarios, val_split, int(seed))
        train_data = [prepare_scenario(s, device) for s in train_raw]
        val_data = [prepare_scenario(s, device) for s in val_raw]

        object_count = train_data[0]["objects"].size(0)
        stages = [p for p in PREFIX_STAGES if p <= object_count] or [object_count]
        state.stages = stages

        state.status = "Precomputing greedy baseline..."
        greedy_costs = []
        with torch.no_grad():
            for s in val_data:
                greedy_costs.append(greedy_rollout(s))
        greedy_mean = float(np.mean(greedy_costs))
        state.greedy_mean = greedy_mean

        state.logs.append(f"Train: {len(train_data)}, Val: {len(val_data)}, Objects: {object_count}")
        state.logs.append(f"Stages: {stages}, Greedy baseline: {greedy_mean:.1f}")

        # Initialize training logger
        dataset_name = extract_dataset_info(dataset_path)
        hyperparams = {
            "epochs_per_stage": int(epochs_per_stage),
            "epochs_final": int(epochs_final),
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "val_split": float(val_split),
            "accumulate_grad": int(accumulate_grad),
            "seed": int(seed),
            "device": str(device),
            "num_train_scenarios": len(train_data),
            "num_val_scenarios": len(val_data),
            "object_count": object_count,
            "stages": stages,
        }
        logger = TrainingLogger(
            dataset_name=dataset_name,
            model_type="gnn",
            base_log_dir="logs",
            hyperparams=hyperparams,
        )
        state.logs.append(f"Logs saved to: {logger.log_dir}")

        torch.manual_seed(int(seed))
        model = GNNPolicy(FEATURE_DIM, int(hidden_dim), int(heads), float(dropout)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                       weight_decay=float(weight_decay), betas=(0.9, 0.95))

        stage_epoch_map = {p: (int(epochs_final) if p == stages[-1] else int(epochs_per_stage)) for p in stages}
        total_epochs = sum(stage_epoch_map.values())
        state.total_epochs = total_epochs

        # Scheduler: count total scenario updates
        def stage_count(prefix):
            ilp = sum(1 for s in train_data if str(prefix) in s["ilp_prefixes"])
            if ilp > 0:
                return ilp
            return sum(1 for s in train_data if len(s.get("greedy_sequence", [])) >= prefix)

        total_batches = max(1, sum(stage_epoch_map[p] * stage_count(p) for p in stages))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches)

        best_gap = -math.inf
        best_state_dict = None

        state.status = "Training..."
        state.running = True

        for prefix in stages:
            if state.stop_requested:
                break

            stage_set = [s for s in train_data if str(prefix) in s["ilp_prefixes"]]
            if not stage_set:
                stage_set = [s for s in train_data if "greedy_sequence" in s and len(s["greedy_sequence"]) >= prefix]
            if not stage_set:
                continue

            state.current_stage = prefix
            if state.stage_transitions and state.stage_transitions[-1] != prefix:
                pass
            state.stage_transitions.append((state.epoch, prefix))
            state.logs.append(f"--- Stage {prefix} ({stage_epoch_map[prefix]} epochs, {len(stage_set)} scenarios) ---")

            for ep in range(stage_epoch_map[prefix]):
                if state.stop_requested:
                    break

                rng = np.random.default_rng(int(seed) + ep + prefix)
                indices = rng.permutation(len(stage_set))

                epoch_losses = []
                epoch_grad_norms = []
                model.train()
                accumulated = 0

                for idx in indices:
                    if state.stop_requested:
                        break
                    loss = compute_scenario_loss(model, stage_set[idx], prefix, device)
                    if loss is None:
                        continue

                    # Gradient accumulation
                    loss = loss / accumulate_grad
                    loss.backward()
                    accumulated += 1

                    if accumulated % accumulate_grad == 0:
                        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        gn_val = float(gn) if isinstance(gn, torch.Tensor) else gn
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        epoch_grad_norms.append(gn_val)

                    state.batch_losses.append(loss.item() * accumulate_grad)  # Store un-accumulated for display
                    state.grad_norms.append(gn_val if accumulated % accumulate_grad == 0 else 0.0)
                    state.learning_rates.append(optimizer.param_groups[0]["lr"])
                    epoch_losses.append(loss.item() * accumulate_grad)

                # Flush remaining gradients
                if accumulated % accumulate_grad != 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    gn_val = float(gn) if isinstance(gn, torch.Tensor) else gn
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    epoch_grad_norms.append(gn_val)

                # Validation
                model.eval()
                metrics = evaluate(model, val_data, device, final_prefix=stages[-1])
                val_mean = metrics["mean_cost"]
                gap = metrics["mean_gap_vs_greedy"]

                state.epoch += 1
                avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                avg_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
                state.train_losses.append(avg_loss)
                state.val_costs.append(val_mean)
                state.gaps.append(gap)

                # Log to training logger
                current_lr = optimizer.param_groups[0]['lr']
                logger.log_epoch(
                    epoch=state.epoch,
                    stage=prefix,
                    train_loss=avg_loss,
                    val_metrics=metrics,
                    learning_rate=current_lr,
                    grad_norm=avg_grad_norm,
                )

                if gap > best_gap:
                    best_gap = gap
                    best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                    state.best_gap = best_gap

                state.status = (f"Stage {prefix} | Epoch {state.epoch}/{total_epochs} | "
                                f"Loss: {avg_loss:.4f} | Gap: {gap:+.2f}% | Best: {best_gap:+.2f}%")
                state.logs.append(f"E{state.epoch} S{prefix} loss={avg_loss:.4f} gap={gap:+.2f}%")

        # Save best model
        if best_state_dict is not None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_path = MODEL_DIR / f"gnn_{object_count}obj_{ts}.pt"
            torch.save({
                "state_dict": best_state_dict,
                "hidden_dim": int(hidden_dim),
                "heads": int(heads),
                "dropout": float(dropout),
                "object_count": object_count,
                "model_type": "gnn",
            }, save_path)
            state.logs.append(f"Saved best model (gap: {best_gap:+.2f}%) to {save_path.name}")
        else:
            state.logs.append("No improvement found.")

        # Generate plots and save logs
        logger.generate_plots()
        logger.save()
        state.logs.append(f"Plots saved to: {logger.log_dir / 'training_curves.png'}")

        state.status = f"Done! Best gap: {best_gap:+.2f}%"

    except Exception as e:
        import traceback
        state.status = f"Error: {e}"
        state.logs.append(traceback.format_exc())
    finally:
        state.running = False


def start_training(dataset, epochs_per_stage, epochs_final, train_limit,
                   hidden_dim, heads, lr, weight_decay, val_split, seed, accumulate_grad):
    if state.running:
        return
    _reset_state()
    state.running = True
    t = threading.Thread(target=train_async, args=(
        dataset, int(epochs_per_stage), int(epochs_final), int(train_limit),
        int(hidden_dim), int(heads), DEFAULT_DROPOUT,
        float(lr), float(weight_decay), float(val_split), int(seed), int(accumulate_grad),
    ), daemon=True)
    t.start()


def stop_training():
    state.stop_requested = True


# ═══════════════════════════════════════════════════════
# TRAINING STATUS + PLOTS
# ═══════════════════════════════════════════════════════

def get_status():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        return state.status, None, None, "\n".join(state.logs[-80:])

    fig1, fig2 = None, None

    if state.val_costs:
        epochs = list(range(1, len(state.val_costs) + 1))

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        ax1.plot(epochs, state.val_costs, "o-", markersize=3, label="GNN (val)", color="#2ecc71")
        ax1.axhline(state.greedy_mean, color="gray", linestyle="--", label=f"Greedy: {state.greedy_mean:.1f}")
        for ep, stg in state.stage_transitions:
            ax1.axvline(ep, color="red", linestyle=":", alpha=0.4)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cost")
        ax1.set_title("Validation Cost")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, state.gaps, "o-", markersize=3, color="#2ecc71")
        ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
        for ep, stg in state.stage_transitions:
            ax2.axvline(ep, color="red", linestyle=":", alpha=0.4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Gap vs Greedy (%)")
        ax2.set_title(f"Gap (best: {state.best_gap:+.2f}%)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    if state.batch_losses:
        fig2, axes = plt.subplots(2, 2, figsize=(14, 7))

        # Loss
        ax = axes[0, 0]
        ax.plot(state.batch_losses, alpha=0.15, color="blue", linewidth=0.5)
        if len(state.batch_losses) > 50:
            w = min(50, len(state.batch_losses) // 4)
            smoothed = np.convolve(state.batch_losses, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, w - 1 + len(smoothed)), smoothed, color="blue", linewidth=1.5, label="Smoothed")
            ax.legend(fontsize=8)
        ax.set_xlabel("Batch")
        ax.set_ylabel("CE Loss")
        ax.set_title("Cross-Entropy Loss")
        ax.grid(True, alpha=0.3)

        # LR
        ax = axes[0, 1]
        ax.plot(state.learning_rates, color="orange")
        ax.set_xlabel("Batch")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Grad norms
        ax = axes[1, 0]
        ax.plot(state.grad_norms, alpha=0.15, color="green", linewidth=0.5)
        if len(state.grad_norms) > 50:
            w = min(50, len(state.grad_norms) // 4)
            smoothed = np.convolve(state.grad_norms, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, w - 1 + len(smoothed)), smoothed, color="green", linewidth=1.5)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Grad Norm")
        ax.set_title("Gradient Norms")
        ax.grid(True, alpha=0.3)

        # Curriculum stages
        ax = axes[1, 1]
        if state.train_losses:
            ax.bar(range(1, len(state.train_losses) + 1), state.train_losses, color="#3498db", alpha=0.7)
            for ep, stg in state.stage_transitions:
                ax.axvline(ep, color="red", linestyle="--", alpha=0.7, label=f"Stage {stg}" if ep == state.stage_transitions[0][0] else "")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean CE Loss")
        ax.set_title(f"Epoch Loss (stage: {state.current_stage})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

    log_text = "\n".join(state.logs[-80:])
    return state.status, fig1, fig2, log_text


# ═══════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════

def run_evaluation(dataset_path, model_path, num_samples, run_ilp=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close("all")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mp = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not mp.exists():
            return f"Model not found: {mp}", None, None, None, ""

        scenarios_raw = load_scenarios(dataset_path)[:int(num_samples)]
        data = [prepare_scenario(s, device) for s in scenarios_raw]
        model = load_model(device, model_path=str(mp))

        gnn_costs, greedy_costs_list, random_costs_list, ilp_costs = [], [], [], []

        with torch.no_grad():
            for i, s in enumerate(data):
                gnn_costs.append(rollout_model(model, s, device))
                greedy_costs_list.append(greedy_rollout(s))
                random_costs_list.append(random_cost(s))
                if run_ilp and ORTOOLS_AVAILABLE:
                    ilp_c, _, _ = solve_ilp_route(scenarios_raw[i])
                    ilp_costs.append(ilp_c)
                else:
                    ilp_costs.append(None)

        gnn_costs = np.array(gnn_costs)
        greedy_costs_arr = np.array(greedy_costs_list)
        random_costs_arr = np.array(random_costs_list)

        per_gap = (greedy_costs_arr - gnn_costs) / greedy_costs_arr * 100
        gap_greedy = per_gap.mean()
        gap_random = ((random_costs_arr - gnn_costs) / random_costs_arr * 100).mean()

        pcts = [5, 10, 25, 50, 75, 90, 95]
        gap_pcts = np.percentile(per_gap, pcts)

        summary = f"""=== EVALUATION ({len(data)} samples) ===

Method         | Mean Cost  | Std Cost   | Min        | Max
---------------|------------|------------|------------|----------
GNN            | {gnn_costs.mean():10.2f} | {gnn_costs.std():10.2f} | {gnn_costs.min():10.2f} | {gnn_costs.max():10.2f}
Greedy         | {greedy_costs_arr.mean():10.2f} | {greedy_costs_arr.std():10.2f} | {greedy_costs_arr.min():10.2f} | {greedy_costs_arr.max():10.2f}
Random         | {random_costs_arr.mean():10.2f} | {random_costs_arr.std():10.2f} | {random_costs_arr.min():10.2f} | {random_costs_arr.max():10.2f}"""

        if run_ilp and any(c is not None for c in ilp_costs):
            valid_ilp = np.array([c for c in ilp_costs if c is not None])
            ilp_gap = (gnn_costs[:len(valid_ilp)] - valid_ilp) / valid_ilp * 100
            summary += f"""
ILP (optimal)  | {valid_ilp.mean():10.2f} | {valid_ilp.std():10.2f} | {valid_ilp.min():10.2f} | {valid_ilp.max():10.2f}"""

        summary += f"""

GNN vs Greedy gap: {gap_greedy:+.2f}% (positive = GNN better)
GNN vs Random gap: {gap_random:+.2f}%

GNN beats Greedy: {(gnn_costs < greedy_costs_arr).sum()}/{len(data)} ({(gnn_costs < greedy_costs_arr).mean()*100:.1f}%)
Greedy beats GNN: {(gnn_costs > greedy_costs_arr).sum()}/{len(data)} ({(gnn_costs > greedy_costs_arr).mean()*100:.1f}%)

=== PERCENTILE BREAKDOWN (Gap vs Greedy %) ===
"""
        for p, v in zip(pcts, gap_pcts):
            summary += f"  P{p:02d}: {v:+.2f}%\n"

        if run_ilp and any(c is not None for c in ilp_costs):
            summary += f"\n=== ILP COMPARISON ===\n"
            summary += f"GNN vs ILP gap: {ilp_gap.mean():+.2f}%\n"
            summary += f"GNN within 1% of ILP: {(np.abs(ilp_gap) < 1.0).sum()}/{len(valid_ilp)}\n"
            summary += f"GNN within 5% of ILP: {(np.abs(ilp_gap) < 5.0).sum()}/{len(valid_ilp)}\n"

        # Bar chart
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        methods = ["GNN", "Greedy", "Random"]
        means = [gnn_costs.mean(), greedy_costs_arr.mean(), random_costs_arr.mean()]
        stds = [gnn_costs.std(), greedy_costs_arr.std(), random_costs_arr.std()]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        if run_ilp and any(c is not None for c in ilp_costs):
            methods.append("ILP")
            means.append(valid_ilp.mean())
            stds.append(valid_ilp.std())
            colors.append("#9b59b6")
        ds_label = Path(dataset_path).stem
        model_label = Path(mp).stem
        ctx = f"{ds_label} | {model_label}"
        bars = ax1.bar(methods, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        ax1.set_ylabel("Mean Cost")
        ax1.set_title(f"Cost Comparison\n{ctx}")
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, f"{mean:.1f}",
                     ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()

        # Gap histogram
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(per_gap, bins=min(40, len(data) // 2 + 1), edgecolor="black", alpha=0.7, color="#2ecc71")
        ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="GNN = Greedy")
        ax2.axvline(gap_greedy, color="darkgreen", linestyle="-", linewidth=2, label=f"Mean: {gap_greedy:+.2f}%")
        ax2.set_xlabel("Gap vs Greedy (%)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Gap Distribution (positive = GNN better)\n{ctx}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        # Scatter
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.scatter(greedy_costs_arr, gnn_costs, alpha=0.5, s=15, color="#3498db", edgecolors="none")
        lims = [min(gnn_costs.min(), greedy_costs_arr.min()), max(gnn_costs.max(), greedy_costs_arr.max())]
        ax3.plot(lims, lims, "r--", linewidth=1.5, label="GNN = Greedy")
        ax3.set_xlabel("Greedy Cost")
        ax3.set_ylabel("GNN Cost")
        ax3.set_title(f"GNN vs Greedy (per scenario)\n{ctx}")
        ax3.set_aspect("equal")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()

        sample_table = "Sample | GNN      | Greedy   | Random   | Gap%     | Best\n"
        sample_table += "-------|----------|----------|----------|----------|------\n"
        for i in range(min(30, len(data))):
            best = "GNN" if gnn_costs[i] <= min(greedy_costs_arr[i], random_costs_arr[i]) else (
                "Greedy" if greedy_costs_arr[i] <= random_costs_arr[i] else "Random")
            if run_ilp and ilp_costs[i] is not None and ilp_costs[i] <= gnn_costs[i]:
                best = "ILP"
            sample_table += f"{i+1:6} | {gnn_costs[i]:8.1f} | {greedy_costs_arr[i]:8.1f} | {random_costs_arr[i]:8.1f} | {per_gap[i]:+7.2f}% | {best}\n"

        return summary, fig1, fig2, fig3, sample_table

    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", None, None, None, ""


# ═══════════════════════════════════════════════════════
# INFERENCE TIMING
# ═══════════════════════════════════════════════════════

def compare_inference_time(dataset_path, model_path, num_samples=10):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close("all")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mp = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not mp.exists():
            return f"Model not found: {mp}", None, ""

        scenarios_raw = load_scenarios(dataset_path)[:int(num_samples)]
        data = [prepare_scenario(s, device) for s in scenarios_raw]
        model = load_model(device, model_path=str(mp))

        # Warmup
        with torch.no_grad():
            if data:
                rollout_model(model, data[0], device)

        all_timings, all_costs = [], []
        with torch.no_grad():
            for s in data:
                cost, timing = rollout_timed(model, s, device)
                all_costs.append(cost)
                all_timings.append(timing)

        total_times = [t["total_time"] for t in all_timings]
        forward_times = [t["model_forward_time"] for t in all_timings]
        graph_times = [t["graph_build_time"] for t in all_timings]

        summary = f"""
=== INFERENCE TIME ({len(data)} samples) ===

Total Rollout:  Mean: {np.mean(total_times)*1000:.2f} ms | Std: {np.std(total_times)*1000:.2f} ms
Model Forward:  Mean: {np.mean(forward_times)*1000:.2f} ms | Std: {np.std(forward_times)*1000:.2f} ms
Graph Build:    Mean: {np.mean(graph_times)*1000:.2f} ms | Std: {np.std(graph_times)*1000:.2f} ms

Time Breakdown:
  Graph build:  {np.mean(graph_times)/np.mean(total_times)*100:.1f}%
  Model forward: {np.mean(forward_times)/np.mean(total_times)*100:.1f}%
  Other:         {(1 - np.mean(graph_times)/np.mean(total_times) - np.mean(forward_times)/np.mean(total_times))*100:.1f}%

Device: {device.type.upper()}
"""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].hist([t * 1000 for t in total_times], bins=min(20, len(total_times)), edgecolor="black", alpha=0.7)
        axes[0, 0].axvline(np.mean(total_times) * 1000, color="red", linestyle="--",
                           label=f"Mean: {np.mean(total_times)*1000:.2f} ms")
        axes[0, 0].set_xlabel("Total Time (ms)")
        axes[0, 0].set_title("Total Rollout Time")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist([t * 1000 for t in forward_times], bins=min(20, len(forward_times)),
                        edgecolor="black", alpha=0.7, color="green")
        axes[0, 1].axvline(np.mean(forward_times) * 1000, color="red", linestyle="--",
                           label=f"Mean: {np.mean(forward_times)*1000:.2f} ms")
        axes[0, 1].set_xlabel("Forward Time (ms)")
        axes[0, 1].set_title("GAT Forward Pass Time")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        if all_timings and all_timings[0].get("per_step_times"):
            step_times_all = [t["per_step_times"] for t in all_timings]
            max_steps = max(len(st) for st in step_times_all)
            step_means, step_stds = [], []
            for si in range(max_steps):
                vals = [st[si] * 1000 for st in step_times_all if si < len(st)]
                step_means.append(np.mean(vals) if vals else 0)
                step_stds.append(np.std(vals) if vals else 0)
            axes[1, 0].plot(range(1, len(step_means) + 1), step_means, marker="o", markersize=3)
            axes[1, 0].fill_between(range(1, len(step_means) + 1),
                                     [m - s for m, s in zip(step_means, step_stds)],
                                     [m + s for m, s in zip(step_means, step_stds)], alpha=0.3)
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Time (ms)")
            axes[1, 0].set_title("Per-Step Time")
            axes[1, 0].grid(True, alpha=0.3)

        if np.mean(total_times) > 0:
            gp = np.mean(graph_times) / np.mean(total_times) * 100
            fp = np.mean(forward_times) / np.mean(total_times) * 100
            op = 100 - gp - fp
            axes[1, 1].pie([gp, fp, op],
                           labels=[f"Graph Build\n{gp:.1f}%", f"GAT Forward\n{fp:.1f}%", f"Other\n{op:.1f}%"],
                           autopct="%1.1f%%", startangle=90)
            axes[1, 1].set_title("Time Breakdown")

        plt.tight_layout()

        table = "Sample | Total (ms) | Forward (ms) | Graph (ms) | Cost\n"
        table += "-------|-----------|--------------|-----------|------\n"
        for i in range(min(20, len(data))):
            table += f"{i+1:6} | {total_times[i]*1000:9.2f} | {forward_times[i]*1000:12.2f} | {graph_times[i]*1000:9.2f} | {all_costs[i]:.1f}\n"

        return summary, fig, table

    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", None, ""


# ═══════════════════════════════════════════════════════
# SCENARIO EXPLORER
# ═══════════════════════════════════════════════════════

def explore_scenario(dataset_path, model_path, scenario_idx, show_gnn, show_greedy, show_ilp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close("all")
        import matplotlib.patches as mpatches
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mp = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        scenarios_raw = load_scenarios(dataset_path)
        idx = int(scenario_idx) - 1
        if idx < 0 or idx >= len(scenarios_raw):
            return f"Invalid index. Dataset has {len(scenarios_raw)} scenarios.", None, ""

        s_raw = scenarios_raw[idx]
        s = prepare_scenario(s_raw, device)
        n = s["objects"].size(0)
        objects_np = np.array(s_raw["objects"])
        bins_np = np.array(s_raw["bins"])
        types_np = np.array(s_raw["types"])
        start_np = np.array(s_raw["start"])

        results = {}
        if show_greedy:
            gc, gr = greedy_rollout_with_route(s)
            results["Greedy"] = (gc, gr)

        if show_gnn:
            if not mp.exists():
                return f"Model not found: {mp.name}\nSelect a valid model.", None, ""
            try:
                model = load_model(device, model_path=str(mp))
            except Exception as e:
                return f"Failed to load model: {e}", None, ""
            gnn_c, gnn_r = gnn_rollout_with_route(model, s, device)
            results["GNN"] = (gnn_c, gnn_r)

        if show_ilp and ORTOOLS_AVAILABLE:
            ilp_c, ilp_r, _ = solve_ilp_route(s_raw)
            if ilp_c is not None:
                results["ILP"] = (ilp_c, ilp_r)

        num_routes = max(1, sum(1 for v in results.values() if v[1] is not None))
        fig, axes = plt.subplots(1, num_routes + 1, figsize=(7 * (num_routes + 1), 7))
        if num_routes + 1 == 1:
            axes = [axes]

        def draw_workspace(ax, title, route=None, route_color="black"):
            ax.set_xlim(-5, WORKSPACE_SIZE + 5)
            ax.set_ylim(-5, WORKSPACE_SIZE + 5)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.15)
            for bi in range(len(bins_np)):
                c = BIN_COLORS[bi % len(BIN_COLORS)]
                ax.plot(bins_np[bi][0], bins_np[bi][1], "s", markersize=18,
                        color=c, markeredgecolor="black", markeredgewidth=2, zorder=5)
                ax.annotate(BIN_NAMES[bi], (bins_np[bi][0], bins_np[bi][1]),
                            textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8, fontweight="bold")
            for oi in range(n):
                c = BIN_COLORS[types_np[oi] % len(BIN_COLORS)]
                ax.plot(objects_np[oi][0], objects_np[oi][1], "o", markersize=8,
                        color=c, markeredgecolor="black", markeredgewidth=0.5, zorder=4)
                ax.annotate(str(oi), (objects_np[oi][0], objects_np[oi][1]),
                            textcoords="offset points", xytext=(4, 4), fontsize=6, color="gray")
            ax.plot(start_np[0], start_np[1], "*", markersize=20, color="gold",
                    markeredgecolor="black", markeredgewidth=1.5, zorder=6)
            ax.annotate("START", (start_np[0], start_np[1]),
                        textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8,
                        fontweight="bold", color="goldenrod")
            if route is not None:
                pos = start_np.copy()
                for step, obj_idx in enumerate(route):
                    obj_p = objects_np[obj_idx]
                    bin_p = bins_np[types_np[obj_idx]]
                    ax.annotate("", xy=obj_p, xytext=pos,
                                arrowprops=dict(arrowstyle="->", color=route_color, lw=1.2, alpha=0.6))
                    ax.annotate("", xy=bin_p, xytext=obj_p,
                                arrowprops=dict(arrowstyle="->", color=route_color, lw=1.2, alpha=0.4, linestyle="dashed"))
                    ax.annotate(str(step + 1), (obj_p[0], obj_p[1]),
                                textcoords="offset points", xytext=(-6, -10), fontsize=6, fontweight="bold",
                                color=route_color,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=route_color, alpha=0.8))
                    pos = bin_p.copy()
            patches = [mpatches.Patch(color=BIN_COLORS[i], label=BIN_NAMES[i]) for i in range(min(NUM_BINS, len(bins_np)))]
            ax.legend(handles=patches, loc="upper right", fontsize=8)

        draw_workspace(axes[0], f"Scenario #{scenario_idx} ({n} objects)")

        route_colors = {"GNN": "#2ecc71", "Greedy": "#3498db", "ILP": "#9b59b6"}
        pi = 1
        for method in ["GNN", "Greedy", "ILP"]:
            if method not in results or results[method][1] is None:
                continue
            cost, route = results[method]
            draw_workspace(axes[pi], f"{method} (cost: {cost:.1f})",
                           route=route, route_color=route_colors.get(method, "black"))
            pi += 1

        plt.tight_layout()

        # Step table
        method_order = [m for m in ["GNN", "Greedy", "ILP"] if m in results and results[m][1] is not None]
        table = "Step | "
        for m in method_order:
            table += f"{m:>8s} Obj | {m:>8s} Type | {m:>8s} Cost | "
        table += "\n" + "-" * (6 + len(method_order) * 38) + "\n"

        max_steps = max((len(results[m][1]) for m in method_order), default=0)
        for si in range(max_steps):
            table += f"{si+1:4} | "
            for m in method_order:
                route = results[m][1]
                if si < len(route):
                    oi = route[si]
                    ti = types_np[oi]
                    prev = start_np if si == 0 else bins_np[types_np[route[si - 1]]]
                    sc = np.linalg.norm(prev - objects_np[oi]) + np.linalg.norm(objects_np[oi] - bins_np[ti])
                    table += f"{oi:>8d}     | {ti:>8d}      | {sc:>10.1f}  | "
                else:
                    table += f"{'':>8s}     | {'':>8s}      | {'':>10s}  | "
            table += "\n"

        table += "-" * (6 + len(method_order) * 38) + "\n"
        table += "Total| "
        for m in method_order:
            table += f"{'':>8s}     | {'':>8s}      | {results[m][0]:>10.1f}  | "
        table += "\n"

        info = f"Scenario {scenario_idx}/{len(scenarios_raw)} | {n} objects | {len(bins_np)} bins\n"
        for m in method_order:
            info += f"  {m}: cost = {results[m][0]:.2f}\n"
        if "GNN" in results and "Greedy" in results and results["GNN"][0] is not None:
            gap = (results["Greedy"][0] - results["GNN"][0]) / results["Greedy"][0] * 100
            info += f"  GNN vs Greedy gap: {gap:+.2f}%\n"
        if "ILP" in results and "GNN" in results and results["ILP"][0] is not None and results["GNN"][0] is not None:
            gap = (results["GNN"][0] - results["ILP"][0]) / results["ILP"][0] * 100
            info += f"  GNN vs ILP gap: {gap:+.2f}%\n"

        return info, fig, table

    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", None, ""


# ═══════════════════════════════════════════════════════
# GRADIO GUI
# ═══════════════════════════════════════════════════════

def _scan_files(directory, ext):
    base = Path(directory)
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob(f"*{ext}"))


def create_gui():
    data_dir = Path(__file__).parent.parent / "data"
    model_dir = Path(__file__).parent.parent / "models"
    datasets = _scan_files(data_dir, ".json")
    models = _scan_files(model_dir, ".pt")
    default_ds = str(data_dir / "dataset_40_objects.json") if datasets else ""
    default_model = str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else (models[0] if models else "")

    with gr.Blocks(title="GAT Training Dashboard") as app:
        gr.Markdown("# GAT Policy — Curriculum Imitation Learning")

        with gr.Tabs():
            # =================== TRAINING ===================
            with gr.TabItem("Train"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        dataset_input = gr.Dropdown(choices=datasets, value=default_ds, label="Dataset",
                                                     allow_custom_value=True)
                        with gr.Row():
                            train_limit_input = gr.Slider(200, 5000, value=5000, step=100, label="Samples")
                            seed_input = gr.Number(value=0, label="Seed", precision=0)
                        with gr.Row():
                            epochs_stage_input = gr.Slider(1, 20, value=DEFAULT_EPOCHS_PER_STAGE, step=1,
                                                            label="Epochs/Stage")
                            epochs_final_input = gr.Slider(1, 30, value=DEFAULT_EPOCHS_FINAL, step=1,
                                                            label="Final Epochs")

                        with gr.Accordion("Architecture", open=False):
                            hidden_dim_input = gr.Slider(64, 512, value=DEFAULT_HIDDEN_DIM, step=64, label="Hidden Dim")
                            num_heads_input = gr.Slider(1, 16, value=DEFAULT_ATTENTION_HEADS, step=1, label="Heads")

                        with gr.Accordion("Optimization", open=False):
                            lr_input = gr.Number(value=DEFAULT_LR, label="Learning Rate")
                            weight_decay_input = gr.Number(value=DEFAULT_WEIGHT_DECAY, label="Weight Decay")
                            accumulate_grad_input = gr.Slider(1, 32, value=DEFAULT_ACCUMULATE_GRAD, step=1, label="Accumulate Grad")
                            val_split_input = gr.Slider(0.05, 0.3, value=DEFAULT_VAL_SPLIT, step=0.05, label="Val Split")

                        with gr.Row():
                            start_btn = gr.Button("Start", variant="primary")
                            stop_btn = gr.Button("Stop", variant="stop")
                            refresh_btn = gr.Button("Refresh")

                    with gr.Column(scale=3):
                        status_text = gr.Textbox(label="Status", lines=3, interactive=False)
                        epoch_plots = gr.Plot(label="Cost + Gap", format="png")
                        batch_plots = gr.Plot(label="Loss / LR / Grad / Stage", format="png")
                        with gr.Accordion("Training Log", open=False):
                            log_box = gr.Textbox(label="Log", lines=15, interactive=False, max_lines=80)

                timer = gr.Timer(value=3, active=False)
                all_train_outputs = [status_text, epoch_plots, batch_plots, log_box]

                def start_and_refresh(*args):
                    try:
                        start_training(*args)
                        time.sleep(0.5)
                        return get_status()
                    except Exception as e:
                        import traceback
                        return f"Error: {e}\n{traceback.format_exc()}", None, None, ""

                start_btn.click(
                    start_and_refresh,
                    inputs=[dataset_input, epochs_stage_input, epochs_final_input, train_limit_input,
                            hidden_dim_input, num_heads_input, lr_input, weight_decay_input,
                            val_split_input, seed_input, accumulate_grad_input],
                    outputs=all_train_outputs,
                ).then(lambda: gr.Timer(active=True), outputs=[timer])

                stop_btn.click(lambda: (stop_training(), gr.Timer(active=False))[-1], outputs=[timer]).then(
                    get_status, outputs=all_train_outputs
                )
                refresh_btn.click(get_status, outputs=all_train_outputs)

                def auto_refresh():
                    result = get_status()
                    if not state.running:
                        return *result, gr.Timer(active=False)
                    return *result, gr.Timer(active=True)

                timer.tick(auto_refresh, outputs=all_train_outputs + [timer])

            # =================== EVALUATION ===================
            with gr.TabItem("Evaluate"):
                with gr.Row():
                    eval_dataset = gr.Dropdown(choices=datasets, value=default_ds, label="Dataset",
                                               allow_custom_value=True)
                    eval_model = gr.Dropdown(choices=models, value=default_model, label="Model",
                                              allow_custom_value=True)
                    eval_refresh = gr.Button("⟳", scale=0, min_width=50)
                    eval_samples = gr.Slider(10, 500, value=100, step=10, label="Samples")
                    eval_ilp = gr.Checkbox(label="Include ILP", value=False)
                    eval_btn = gr.Button("Run", variant="primary")

                eval_summary = gr.Textbox(label="Summary", lines=20, interactive=False)
                with gr.Row():
                    eval_bar = gr.Plot(label="Cost Comparison", format="png")
                    eval_hist = gr.Plot(label="Gap Distribution", format="png")
                eval_scatter = gr.Plot(label="GNN vs Greedy Scatter", format="png")
                with gr.Accordion("Per-Sample Table", open=False):
                    eval_table = gr.Textbox(label="Rows", lines=35, interactive=False)

                eval_refresh.click(lambda: gr.Dropdown(choices=_scan_files(model_dir, ".pt")), outputs=[eval_model])
                eval_btn.click(run_evaluation, inputs=[eval_dataset, eval_model, eval_samples, eval_ilp],
                               outputs=[eval_summary, eval_bar, eval_hist, eval_scatter, eval_table])

                with gr.Accordion("Inference Time Profiling", open=False):
                    with gr.Row():
                        time_dataset = gr.Dropdown(choices=datasets, value=default_ds, label="Dataset",
                                                    allow_custom_value=True)
                        time_model = gr.Dropdown(choices=models, value=default_model, label="Model",
                                                  allow_custom_value=True)
                        time_refresh = gr.Button("⟳", scale=0, min_width=50)
                        time_samples = gr.Slider(5, 100, value=10, step=5, label="Samples")
                        time_btn = gr.Button("Profile", variant="primary")
                    time_summary = gr.Textbox(label="Timing", lines=18, interactive=False)
                    time_plot = gr.Plot(label="Timing Analysis", format="png")
                    time_table = gr.Textbox(label="Per-Sample", lines=20, interactive=False)
                    time_refresh.click(lambda: gr.Dropdown(choices=_scan_files(model_dir, ".pt")), outputs=[time_model])
                    time_btn.click(compare_inference_time, inputs=[time_dataset, time_model, time_samples],
                                   outputs=[time_summary, time_plot, time_table])

            # =================== EXPLORER ===================
            with gr.TabItem("Explorer"):
                with gr.Row():
                    explore_dataset = gr.Dropdown(choices=datasets, value=default_ds, label="Dataset",
                                                   allow_custom_value=True)
                    explore_model = gr.Dropdown(choices=models, value=default_model, label="Model",
                                                 allow_custom_value=True)
                    explore_refresh = gr.Button("⟳", scale=0, min_width=50)
                    explore_idx = gr.Slider(1, 1000, value=1, step=1, label="Scenario #")
                    explore_gnn = gr.Checkbox(label="GNN", value=True)
                    explore_greedy = gr.Checkbox(label="Greedy", value=True)
                    explore_ilp = gr.Checkbox(label="ILP", value=False)
                    explore_btn = gr.Button("Go", variant="primary")

                explore_info = gr.Textbox(label="Info", lines=5, interactive=False)
                explore_plot = gr.Plot(label="Workspace", format="png")
                with gr.Accordion("Step-by-Step", open=False):
                    explore_table = gr.Textbox(label="Route", lines=45, interactive=False)

                explore_refresh.click(lambda: gr.Dropdown(choices=_scan_files(model_dir, ".pt")), outputs=[explore_model])
                explore_btn.click(explore_scenario,
                                  inputs=[explore_dataset, explore_model, explore_idx,
                                          explore_gnn, explore_greedy, explore_ilp],
                                  outputs=[explore_info, explore_plot, explore_table])

    return app


if __name__ == "__main__":
    import os
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app = create_gui()
    app.launch(share=False, server_name="0.0.0.0", server_port=port)
