"""
Heterogeneous GNN variant for PAP sequencing.

Uses torch_geometric.data.HeteroData with three node types (robot, object, bin)
and three typed directed edge relations:
  - (robot, pick,     object)
  - (object, place,   bin)
  - (bin,   feedback, robot)

Unlike gnn_train.GNNPolicy which operates on a homogeneous GATConv with all
nodes concatenated into one tensor and edge types encoded only in the node
features, this module uses PyG's HeteroConv so each relation owns its own
attention weights.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

from gnn_train import (
    FEATURE_DIM,
    MASK_FILL_VALUE,
    NUM_BINS,
    WORKSPACE_SIZE,
)


OBJECT_FEATURE_DIM = 2 + 2 + NUM_BINS + 1          # pos + bin pos + onehot + mask
BIN_FEATURE_DIM = 2 + NUM_BINS                      # pos + identity one-hot
ROBOT_FEATURE_DIM = 2                                # pos


def build_hetero_step_graph(
    objects: torch.Tensor,
    bins: torch.Tensor,
    types: torch.Tensor,
    mask: torch.Tensor,
    robot_world: torch.Tensor,
    device: torch.device,
) -> Tuple[HeteroData, torch.Tensor]:
    n_objects = objects.size(0)
    robot_norm = robot_world / WORKSPACE_SIZE
    obj_norm = objects / WORKSPACE_SIZE
    bin_norm = bins / WORKSPACE_SIZE
    type_one_hot = F.one_hot(types, num_classes=NUM_BINS).float()

    robot_feat = torch.zeros((1, ROBOT_FEATURE_DIM), device=device)
    robot_feat[0] = robot_norm

    obj_feat = torch.cat(
        [
            obj_norm,
            bin_norm[types],
            type_one_hot,
            mask.float().unsqueeze(-1),
        ],
        dim=-1,
    )

    bin_feat = torch.cat([bin_norm, torch.eye(NUM_BINS, device=device)], dim=-1)

    data = HeteroData()
    data["robot"].x = robot_feat
    data["object"].x = obj_feat
    data["bin"].x = bin_feat

    valid_obj = torch.nonzero(mask, as_tuple=False).flatten()
    if valid_obj.numel() > 0:
        src_r = torch.zeros(valid_obj.size(0), dtype=torch.long, device=device)
        data["robot", "pick", "object"].edge_index = torch.stack([src_r, valid_obj], dim=0)

        tgt_b = types[valid_obj]
        data["object", "place", "bin"].edge_index = torch.stack([valid_obj, tgt_b], dim=0)

        unique_bins = torch.unique(tgt_b)
        dst_r = torch.zeros(unique_bins.size(0), dtype=torch.long, device=device)
        data["bin", "feedback", "robot"].edge_index = torch.stack([unique_bins, dst_r], dim=0)
    else:
        empty = torch.zeros((2, 0), dtype=torch.long, device=device)
        data["robot", "pick", "object"].edge_index = empty
        data["object", "place", "bin"].edge_index = empty
        data["bin", "feedback", "robot"].edge_index = empty

    return data, mask


class HeteroGNNPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()

        def make_conv():
            return HeteroConv(
                {
                    ("robot", "pick", "object"): GATConv(
                        (-1, -1), hidden_dim, heads=heads, concat=False,
                        dropout=dropout, add_self_loops=False,
                    ),
                    ("object", "place", "bin"): GATConv(
                        (-1, -1), hidden_dim, heads=heads, concat=False,
                        dropout=dropout, add_self_loops=False,
                    ),
                    ("bin", "feedback", "robot"): GATConv(
                        (-1, -1), hidden_dim, heads=heads, concat=False,
                        dropout=dropout, add_self_loops=False,
                    ),
                },
                aggr="sum",
            )

        self.conv1 = make_conv()
        self.conv2 = make_conv()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: HeteroData, obj_mask: torch.Tensor) -> torch.Tensor:
        x_dict = {k: v for k, v in data.x_dict.items()}
        edge_index_dict = data.edge_index_dict

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        h_obj = x_dict["object"]
        logits = self.head(h_obj).squeeze(-1)
        logits = logits.masked_fill(~obj_mask, MASK_FILL_VALUE)
        return logits


def compute_scenario_loss_hetero(
    model: HeteroGNNPolicy,
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
        data, obj_mask = build_hetero_step_graph(
            objects, bins, types, mask, robot_world, device
        )
        logits = model(data, obj_mask)
        target_tensor = torch.tensor([target], device=device, dtype=torch.long)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target_tensor))
        robot_world = bins[types[target]]
        mask[target] = False

    if not losses:
        return None
    return torch.stack(losses).mean()


def rollout_hetero(
    model: HeteroGNNPolicy, scenario: Dict, device: torch.device
) -> float:
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    mask = torch.ones(objects.size(0), device=device, dtype=torch.bool)
    robot_world = scenario["start"]
    total = 0.0
    steps = 0
    while mask.any():
        data, obj_mask = build_hetero_step_graph(
            objects, bins, types, mask, robot_world, device
        )
        logits = model(data, obj_mask)
        action = int(torch.argmax(logits).item())
        if not mask[action]:
            break
        obj_pos = objects[action]
        bin_pos = bins[types[action]]
        total += torch.norm(robot_world - obj_pos).item()
        total += torch.norm(obj_pos - bin_pos).item()
        robot_world = bin_pos
        mask[action] = False
        steps += 1
        if steps > objects.size(0) + 1:
            break
    return total
