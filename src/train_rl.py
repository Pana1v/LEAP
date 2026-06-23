import json
import math
import numpy as np
import os
import random
import torch
from typing import List, Dict, Optional, Any
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from binning_env import BinningEnv
from tqdm import tqdm
import argparse


def load_scenarios(dataset_path: str, max_scenarios: int = None) -> List[Dict]:
    """Load scenarios from dataset file."""
    with open(dataset_path, 'r') as f:
        scenarios = json.load(f)
    
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]
    
    return scenarios


def infer_max_objects(scenarios: List[Dict], override: Optional[int]) -> int:
    """Determine max_objects to match scenario sizes and avoid invalid actions."""
    dataset_max = max(len(s["objects"]) for s in scenarios)
    if override is not None:
        return min(override, dataset_max)
    return dataset_max


def create_env_factory(scenarios: List[Dict], max_objects: int, seed: int = None, wrap_monitor: bool = False):
    """Create environment factory for vectorized environments with per-episode resampling."""
    base_rng = np.random.default_rng(seed)
    
    def make_env():
        # Give each environment its own RNG to avoid correlated sampling
        env_rng = np.random.default_rng(base_rng.integers(1_000_000_000))

        def sample_scenario():
            return scenarios[env_rng.integers(len(scenarios))]
        
        # Each environment resamples a scenario on every reset via the sampler
        env = BinningEnv(
            scenario=sample_scenario(),
            max_objects=max_objects,
            scenario_sampler=sample_scenario
        )
        if wrap_monitor:
            env = Monitor(env)

        # Wrap with action masking to prevent already-picked actions
        def mask_fn(env_instance):
            base_env = getattr(env_instance, "env", env_instance)
            return base_env.get_action_mask()
        return ActionMasker(env, mask_fn)
    
    return make_env


EVAL_FREQUENCY_STEPS = 10_000
CHECKPOINT_FREQUENCY_STEPS = 50_000
INVALID_RUN_MULTIPLIER = 2.0  # fallback cost multiplier when rollout fails
GREEDY_PLOT_SAMPLE = 25  # number of eval scenarios to plot against greedy
EVAL_EPOCH_INTERVAL = 5  # evaluate vs greedy & random every N epochs


class TQDMProgressCallback(BaseCallback):
    """Progress bar over PPO rollout epochs using tqdm."""

    def __init__(self, total_timesteps: int, epoch_size: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.epoch_size = max(1, epoch_size)
        self.total_epochs = max(1, math.ceil(total_timesteps / self.epoch_size))
        self.pbar = None
        self.epoch = 0

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_epochs, desc="Training epochs", leave=True)

    def _on_rollout_end(self) -> bool:
        if self.pbar:
            self.epoch += 1
            self.pbar.update(1)
            self.pbar.set_postfix(timesteps=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

    def _on_step(self) -> bool:
        # Required abstract method; keep stepping
        return True


def _evaluate_mean_cost(model: Any, scenarios: List[Dict], max_objects: int) -> float:
    """Compute mean cost for the current policy across provided scenarios."""
    costs = []
    for scenario in scenarios:
        env = BinningEnv(scenario, max_objects=max_objects)
        obs, info = env.reset()
        done = False
        invalid_action = False
        truncated = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        invalid_action = info.get("invalid_action", False)
        success = (not truncated) and (not invalid_action) and len(env.remaining_objects) == 0
        rl_cost = env.total_distance if success else scenario["greedy_cost"] * INVALID_RUN_MULTIPLIER
        costs.append(rl_cost)

    return float(np.mean(costs)) if costs else 0.0


def _sequence_cost(sequence: List[int], scenario: Dict) -> float:
    """Compute exact cost of a provided sequence for a scenario."""
    objects = np.array(scenario["objects"], dtype=np.float32)
    bins = np.array(scenario["bins"], dtype=np.float32)
    types = np.array(scenario["types"], dtype=np.int32)
    start = np.array(scenario["start"], dtype=np.float32)

    cost = 0.0
    current = start.copy()
    for idx in sequence:
        obj = objects[idx]
        bin_pos = bins[types[idx]]
        cost += np.linalg.norm(current - obj)
        cost += np.linalg.norm(obj - bin_pos)
        current = bin_pos
    return float(cost)


def _evaluate_random_mean_cost(scenarios: List[Dict]) -> float:
    """Compute mean cost of random permutations for each scenario."""
    costs = []
    for scenario in scenarios:
        n = len(scenario["objects"])
        seq = np.random.permutation(n).tolist()
        costs.append(_sequence_cost(seq, scenario))
    return float(np.mean(costs)) if costs else 0.0


class GreedyComparisonCallback(BaseCallback):
    """
    Logs mean RL cost, greedy baseline cost, and random baseline cost to TensorBoard,
    and prints margins every EVAL_EPOCH_INTERVAL epochs.
    """

    def __init__(self, eval_scenarios: List[Dict], max_objects: int, eval_freq: int, epoch_interval: int, epoch_size: int):
        super().__init__(verbose=0)
        self.max_objects = max_objects
        self.eval_freq = max(1, eval_freq)
        self.epoch_interval = max(1, epoch_interval)
        self.epoch_size = max(1, epoch_size)
        # Limit eval set for speed while keeping a stable signal
        self.eval_scenarios = eval_scenarios[:GREEDY_PLOT_SAMPLE] if eval_scenarios else []
        self.greedy_mean_cost = (
            float(np.mean([s["greedy_cost"] for s in self.eval_scenarios])) if self.eval_scenarios else 0.0
        )
        self.eval_counter = 0
        self.rollout_counter = 0
        self.random_mean_cost = _evaluate_random_mean_cost(self.eval_scenarios) if self.eval_scenarios else 0.0

    def _on_training_start(self) -> None:
        # Seed the greedy line so it is visible from the first evaluation step
        if self.logger is not None:
            self.logger.record("comparison/greedy_mean_cost", self.greedy_mean_cost)
            self.logger.record("comparison/random_mean_cost", self.random_mean_cost)

    def _on_step(self) -> bool:
        # Align plotting with eval frequency to keep overhead predictable
        if self.n_calls % self.eval_freq != 0:
            return True

        rl_mean_cost = _evaluate_mean_cost(self.model, self.eval_scenarios, self.max_objects)
        self.eval_counter += 1

        # Log to TensorBoard; users can view RL vs greedy in the same GUI
        if self.logger is not None:
            self.logger.record("comparison/rl_mean_cost", rl_mean_cost)
            self.logger.record("comparison/greedy_mean_cost", self.greedy_mean_cost)
            self.logger.record("comparison/random_mean_cost", self.random_mean_cost)
            self.logger.record("comparison/eval_step", self.eval_counter)

        # Every epoch_interval rollouts, print margins vs greedy and random
        # Rollouts counted via n_calls / epoch_size approximations
        self.rollout_counter = self.num_timesteps // self.epoch_size
        if self.rollout_counter % self.epoch_interval == 0:
            greedy_margin = ((self.greedy_mean_cost - rl_mean_cost) / self.greedy_mean_cost * 100.0) if self.greedy_mean_cost else 0.0
            random_margin = ((self.random_mean_cost - rl_mean_cost) / self.random_mean_cost * 100.0) if self.random_mean_cost else 0.0
            print(f"[Eval@epoch {self.rollout_counter}] RL mean cost: {rl_mean_cost:.2f} | Greedy: {self.greedy_mean_cost:.2f} ({greedy_margin:+.2f}%) | Random: {self.random_mean_cost:.2f} ({random_margin:+.2f}%)")

        return True


def train_model(
    dataset_path: str,
    object_count: int,
    max_scenarios: int = 1000,
    total_timesteps: Optional[int] = None,
    n_envs: int = 4,
    max_objects: Optional[int] = None,
    model_save_path: str = None,
    seed: int = 0,
    epochs: Optional[int] = None,
    n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    ppo_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    clip_range: Optional[float] = None,
    ent_coef: Optional[float] = None,
    gamma: Optional[float] = None,
    gae_lambda: Optional[float] = None,
    load_path: Optional[str] = None
):
    """Train RL model on binning dataset."""
    
    print(f"Loading dataset: {dataset_path}")
    scenarios = load_scenarios(dataset_path, max_scenarios)
    print(f"Loaded {len(scenarios)} scenarios")

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    resolved_max_objects = infer_max_objects(scenarios, max_objects)
    print(f"Using max_objects={resolved_max_objects} (inferred from dataset)")
    
    # Create vectorized environment
    print(f"Creating {n_envs} parallel environments...")
    env_factory = create_env_factory(scenarios, resolved_max_objects, seed=seed, wrap_monitor=False)
    env = make_vec_env(env_factory, n_envs=n_envs, seed=seed)
    
    # Create evaluation environment
    eval_scenarios = scenarios[:min(100, len(scenarios))]  # Use first 100 for eval
    eval_env_factory = create_env_factory(
        eval_scenarios,
        resolved_max_objects,
        seed=seed + 1 if seed is not None else None,
        wrap_monitor=True
    )
    eval_env = DummyVecEnv([eval_env_factory])
    
    # Model save path
    if model_save_path is None:
        model_save_path = f"../models/rl_model_{object_count}objects"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Create PPO model
    print("Creating PPO model...")
    try:
        import tensorboard
        tensorboard_log = f"{model_save_path}_tensorboard"
    except ImportError:
        tensorboard_log = None
        print("Tensorboard not available, skipping tensorboard logging")
    
    ppo_n_steps = n_steps if n_steps is not None else 256  # fewer steps per epoch
    ppo_batch_size = batch_size if batch_size is not None else 64
    ppo_n_epochs = ppo_epochs if ppo_epochs is not None else 10
    ppo_lr = learning_rate if learning_rate is not None else 3e-4
    ppo_clip = clip_range if clip_range is not None else 0.2
    ppo_ent = ent_coef if ent_coef is not None else 0.01
    ppo_gamma = gamma if gamma is not None else 0.99
    ppo_gae_lambda = gae_lambda if gae_lambda is not None else 0.95

    epoch_size = ppo_n_steps * n_envs  # steps per PPO rollout across envs
    # Derive total_timesteps from epochs if requested
    if epochs is not None:
        total_timesteps = epoch_size * epochs
    if total_timesteps is None:
        total_timesteps = 1000000  # fallback default
        print(f"total_timesteps not provided; defaulting to {total_timesteps}")
    else:
        print(f"Training for {total_timesteps} timesteps (epochs request={epochs})")
    progress_callback = TQDMProgressCallback(total_timesteps, epoch_size)

    # Create callbacks (after epoch_size is defined)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_save_path}_best",
        log_path=f"{model_save_path}_logs",
        eval_freq=EVAL_FREQUENCY_STEPS,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY_STEPS,
        save_path=f"{model_save_path}_checkpoints",
        name_prefix="rl_model"
    )
    greedy_comparison_callback = GreedyComparisonCallback(
        eval_scenarios=eval_scenarios,
        max_objects=resolved_max_objects,
        eval_freq=EVAL_FREQUENCY_STEPS,
        epoch_interval=EVAL_EPOCH_INTERVAL,
        epoch_size=epoch_size
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if load_path:
        print(f"Loading model from checkpoint: {load_path}")
        model = MaskablePPO.load(
            load_path,
            env=env,
            device=device
        )
        # Ensure env and tensorboard settings are aligned after load
        model.set_env(env)
        if tensorboard_log:
            model.tensorboard_log = tensorboard_log
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            learning_rate=ppo_lr,
            n_steps=ppo_n_steps,
            batch_size=ppo_batch_size,
            n_epochs=ppo_n_epochs,
            gamma=ppo_gamma,
            gae_lambda=ppo_gae_lambda,
            clip_range=ppo_clip,
            ent_coef=ppo_ent,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device
        )
    
    # Train model
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, progress_callback, greedy_comparison_callback],
        progress_bar=False  # Use custom tqdm callback instead of SB3 progress bar
    )
    
    # Save final model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL model for binning task")
    parser.add_argument(
        "--dataset",
        type=str,
        default="../data/dataset_10_objects.json",
        help="Path to dataset file"
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=1000,
        help="Maximum number of scenarios to use for training"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides --epochs if provided)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of PPO rollout epochs (overrides timesteps when set)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Number of steps per rollout per environment (PPO n_steps)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for PPO updates"
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=None,
        help="Number of optimization epochs per PPO update"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for PPO"
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=None,
        help="PPO clip range"
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="Entropy coefficient"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help="GAE lambda"
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint to resume training"
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Maximum number of objects (state space); if omitted, inferred from dataset"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Extract object count from dataset path
    object_count = None
    for count in [3, 5, 10, 40, 100, 200]:
        if f"{count}_objects" in args.dataset:
            object_count = count
            break
    
    if object_count is None:
        object_count = 10  # Default
    
    train_model(
        dataset_path=args.dataset,
        object_count=object_count,
        max_scenarios=args.max_scenarios,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        max_objects=args.max_objects,
        model_save_path=args.model_path,
        seed=args.seed,
        epochs=args.epochs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        load_path=args.load_path
    )


if __name__ == "__main__":
    main()

