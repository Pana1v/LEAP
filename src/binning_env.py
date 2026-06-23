import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Callable
import json

# Reward configuration to control magnitudes
DIST_SCALE = 10.0               # distance divisor; smaller -> larger |reward|
INVALID_ACTION_PENALTY = -5.0   # penalty for invalid picks
END_ON_INVALID = False          # keep episode alive on invalid actions to avoid tiny ep_len
COMPLETION_BONUS_SCALE = 5.0    # scales improvement bonus at episode end


class BinningEnv(gym.Env):
    """
    Gymnasium environment for pick-and-place binning task.
    
    State: Robot position (2D) + remaining objects (positions + types) flattened
    Action: Index of object to pick next (discrete)
    Reward: Negative distance traveled (to minimize cost)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        scenario: Dict,
        max_objects: int = 200,
        scenario_sampler: Optional[Callable[[], Dict]] = None
    ):
        super().__init__()
        
        self.max_objects = max_objects
        self.workspace_size = 100.0
        self.scenario_sampler = scenario_sampler
        
        # Load initial scenario data
        self._load_scenario(scenario)
        
        # State space: robot_pos (2) + remaining_mask (max_objects) + 
        #              object_positions (max_objects * 2) + object_types (max_objects)
        # Total: 2 + max_objects + max_objects * 2 + max_objects = 2 + 4 * max_objects
        state_dim = 2 + 4 * self.max_objects
        
        # Action space: which object to pick (discrete, max_objects)
        self.action_space = spaces.Discrete(self.max_objects)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        
        self.reset()

    def _load_scenario(self, scenario: Dict) -> None:
        """Load scenario data into environment state."""
        self.scenario = scenario
        
        self.objects = np.array(scenario["objects"], dtype=np.float32)
        self.types = np.array(scenario["types"], dtype=np.int32)
        self.bins = np.array(scenario["bins"], dtype=np.float32)
        self.start_pos = np.array(scenario["start"], dtype=np.float32)
        self.greedy_cost = scenario["greedy_cost"]
        
        self.num_objects = len(self.objects)
        self.num_bins = len(self.bins)
        
        # Normalize positions to [0, 1] range (workspace is 100x100)
        self.objects_norm = self.objects / self.workspace_size
        self.bins_norm = self.bins / self.workspace_size
        self.start_pos_norm = self.start_pos / self.workspace_size
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Optionally resample a scenario each episode
        if self.scenario_sampler is not None:
            self._load_scenario(self.scenario_sampler())
        
        # Initialize state
        self.robot_pos = self.start_pos_norm.copy()
        self.remaining_objects = set(range(self.num_objects))
        self.total_distance = 0.0
        self.step_count = 0
        
        observation = self._get_observation()
        info = {"remaining_count": len(self.remaining_objects)}
        
        return observation, info

    def get_action_mask(self) -> np.ndarray:
        """
        Return a boolean mask over actions where True means action is allowed.
        Only remaining object indices are valid.
        """
        mask = np.zeros(self.max_objects, dtype=bool)
        for idx in self.remaining_objects:
            if idx < self.max_objects:
                mask[idx] = True
        return mask
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs = np.zeros(2 + 4 * self.max_objects, dtype=np.float32)
        
        # Robot position (normalized)
        obs[0:2] = self.robot_pos
        
        # Remaining objects mask and data
        remaining_list = sorted(self.remaining_objects)
        
        for i, obj_idx in enumerate(remaining_list):
            if i >= self.max_objects:
                break
            
            # Mask: 1 if object exists
            obs[2 + i] = 1.0
            
            # Object position (normalized)
            obs[2 + self.max_objects + 2 * i] = self.objects_norm[obj_idx, 0]
            obs[2 + self.max_objects + 2 * i + 1] = self.objects_norm[obj_idx, 1]
            
            # Object type (normalized to [0, 1])
            obs[2 + 3 * self.max_objects + i] = self.types[obj_idx] / (self.num_bins - 1)
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Check if action is valid
        if action not in self.remaining_objects:
            # Invalid action: penalty, optionally continue episode
            reward = INVALID_ACTION_PENALTY
            terminated = END_ON_INVALID
            truncated = False
            observation = self._get_observation()
            info = {
                "remaining_count": len(self.remaining_objects),
                "invalid_action": True,
                "total_distance": self.total_distance * self.workspace_size
            }
            return observation, reward, terminated, truncated, info
        
        # Calculate distances
        obj_pos = self.objects[action]
        obj_pos_norm = self.objects_norm[action]
        bin_idx = self.types[action]
        bin_pos = self.bins[bin_idx]
        bin_pos_norm = self.bins_norm[bin_idx]
        
        # Distance from robot to object
        robot_pos_actual = self.robot_pos * self.workspace_size
        dist_pick = np.linalg.norm(robot_pos_actual - obj_pos)
        
        # Distance from object to bin
        dist_place = np.linalg.norm(obj_pos - bin_pos)
        
        total_dist = dist_pick + dist_place
        self.total_distance += total_dist
        
        # Update state
        self.remaining_objects.discard(action)
        self.robot_pos = bin_pos_norm.copy()
        
        # Reward: negative distance (we want to minimize total distance)
        # Scaled to boost signal magnitude
        reward = -total_dist / DIST_SCALE
        
        # Check if done
        terminated = len(self.remaining_objects) == 0
        truncated = False
        
        # Bonus reward for completion
        if terminated:
            # Compare to greedy cost
            improvement = (self.greedy_cost - self.total_distance)
            reward += COMPLETION_BONUS_SCALE * max(0.0, improvement / DIST_SCALE)  # scaled bonus
        
        observation = self._get_observation()
        info = {
            "remaining_count": len(self.remaining_objects),
            "total_distance": self.total_distance,
            "greedy_cost": self.greedy_cost,
            "improvement": (self.greedy_cost - self.total_distance) / self.greedy_cost if terminated else None
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (optional)."""
        pass
    
    def get_sequence_cost(self, sequence: List[int]) -> float:
        """Calculate cost for a given sequence."""
        cost = 0.0
        current_pos = self.start_pos.copy()
        
        for obj_idx in sequence:
            obj_pos = self.objects[obj_idx]
            bin_pos = self.bins[self.types[obj_idx]]
            
            cost += np.linalg.norm(current_pos - obj_pos)
            cost += np.linalg.norm(obj_pos - bin_pos)
            current_pos = bin_pos
        
        return cost

