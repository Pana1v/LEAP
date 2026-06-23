import json
import numpy as np
from typing import List, Dict
from stable_baselines3 import PPO
from binning_env import BinningEnv
import argparse
from tqdm import tqdm


INVALID_RUN_MULTIPLIER = 2.0  # cost fallback multiplier for invalid/failed runs


def evaluate_model(
    model_path: str,
    dataset_path: str,
    num_scenarios: int = 100,
    max_objects: int = None
):
    """Evaluate RL model against greedy baseline."""
    
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        scenarios = json.load(f)
    
    scenarios = scenarios[:num_scenarios]
    if max_objects is None:
        max_objects = max(len(s["objects"]) for s in scenarios)
    print(f"Evaluating on {len(scenarios)} scenarios")
    
    rl_costs = []
    greedy_costs = []
    improvements = []
    
    for scenario in tqdm(scenarios, desc="Evaluating"):
        env = BinningEnv(scenario, max_objects=max_objects)
        
        # RL solution
        obs, info = env.reset()
        done = False
        invalid_action = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        
        # Calculate RL cost
        invalid_action = info.get("invalid_action", False)
        success = (not truncated) and (not invalid_action) and len(env.remaining_objects) == 0
        rl_cost = env.total_distance if success else env.greedy_cost * INVALID_RUN_MULTIPLIER
        
        greedy_cost = scenario["greedy_cost"]
        
        rl_costs.append(rl_cost)
        greedy_costs.append(greedy_cost)
        
        improvement = (greedy_cost - rl_cost) / greedy_cost * 100
        improvements.append(improvement)
    
    # Statistics
    rl_costs = np.array(rl_costs)
    greedy_costs = np.array(greedy_costs)
    improvements = np.array(improvements)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of scenarios: {len(scenarios)}")
    print(f"\nGreedy Baseline:")
    print(f"  Mean cost: {np.mean(greedy_costs):.2f}")
    print(f"  Std cost: {np.std(greedy_costs):.2f}")
    print(f"\nRL Model:")
    print(f"  Mean cost: {np.mean(rl_costs):.2f}")
    print(f"  Std cost: {np.std(rl_costs):.2f}")
    print(f"\nImprovement:")
    print(f"  Mean improvement: {np.mean(improvements):.2f}%")
    print(f"  Std improvement: {np.std(improvements):.2f}%")
    print(f"  Best improvement: {np.max(improvements):.2f}%")
    print(f"  Worst improvement: {np.min(improvements):.2f}%")
    print(f"  Scenarios better than greedy: {np.sum(improvements > 0)}/{len(scenarios)} ({np.sum(improvements > 0)/len(scenarios)*100:.1f}%)")
    print(f"  Average cost reduction: {np.mean(greedy_costs - rl_costs):.2f}")
    print("=" * 60)
    
    return {
        "rl_costs": rl_costs.tolist(),
        "greedy_costs": greedy_costs.tolist(),
        "improvements": improvements.tolist(),
        "mean_improvement": float(np.mean(improvements)),
        "scenarios_better": int(np.sum(improvements > 0))
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file"
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=100,
        help="Number of scenarios to evaluate"
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=200,
        help="Maximum number of objects"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        dataset_path=args.dataset,
        num_scenarios=args.num_scenarios,
        max_objects=args.max_objects
    )


if __name__ == "__main__":
    main()

