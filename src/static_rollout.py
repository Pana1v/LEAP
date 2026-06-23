"""One-shot static arc scorer — drop-in replacement for the autoregressive
GNN rollout, with O(1) forward passes instead of O(N).

Insight: the policy's outgoing-arc preference from object i depends on i only
through the robot position after placing it, which is bins[type[i]]. With
NUM_BINS bins plus the start, there are only NUM_BINS+1 distinct robot
positions ever. So instead of N sequential masked forward passes, run exactly
NUM_BINS+1 full-mask passes (one per robot position) and reuse the scores.

This captures the robot-position dependence EXACTLY and only approximates the
mask-shrinkage effect (scores computed with all objects present). The greedy
decode and the per-step logits are assembled so that downstream
`_rank_outgoing_arcs` / `solve_circuit_pruned` consume them unchanged.

Drop-in for gnn_ilp_circuit.gnn_rollout_with_logits:
    static_rollout_with_logits(model, scenario, device) -> (cost, route, logits_per_step)
"""
import numpy as np
import torch

from gnn_train import _build_step_graph, NUM_BINS


def static_rollout_with_logits(model, scenario, device, keep_prob=1.0,
                               n_samples=1, rng_seed=0):
    """keep_prob<1 scores the bin positions with partial (mid-tour-like) masks,
    averaging each object's logit over `n_samples` masks in which it is present.
    keep_prob=1 (default) reproduces the plain full-mask 5-pass scorer. The
    'start' position is always scored at full mask (it is in-distribution)."""
    model.eval()
    objects = scenario["objects"]
    bins = scenario["bins"]
    types = scenario["types"]
    start = scenario["start"]
    n = objects.size(0)

    full_mask = torch.ones(n, dtype=torch.bool, device=device)

    def score_full(robot_world):
        nf, ei, om = _build_step_graph(objects, bins, types, full_mask, robot_world, device)
        return model(nf, ei, om).detach().cpu().numpy()

    def score_partial(robot_world, gen):
        # average each object's logit over samples where it is kept present
        acc = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        for _ in range(n_samples):
            keep = torch.rand(n, generator=gen, device="cpu") < keep_prob
            if not keep.any():
                keep[0] = True
            m = keep.to(device)
            nf, ei, om = _build_step_graph(objects, bins, types, m, robot_world, device)
            lg = model(nf, ei, om).detach().cpu().numpy()
            kn = keep.numpy()
            acc[kn] += lg[kn]
            cnt[kn] += 1
        cnt[cnt == 0] = 1
        return acc / cnt

    gen = torch.Generator(device="cpu").manual_seed(rng_seed)
    # key 'start' -> robot at start; key t (0..NUM_BINS-1) -> robot at bins[t].
    static_logits = {"start": score_full(start)}
    with torch.no_grad():
        for t in range(NUM_BINS):
            if keep_prob >= 1.0:
                static_logits[t] = score_full(bins[t])
            else:
                static_logits[t] = score_partial(bins[t], gen)

    # Greedy decode over the static scores. At each step the current robot
    # position selects which score vector to use; mask out visited objects.
    obj_np = objects.detach().cpu().numpy()
    bin_np = bins.detach().cpu().numpy()
    type_np = types.detach().cpu().numpy()
    start_np = start.detach().cpu().numpy()

    visited = np.zeros(n, dtype=bool)
    route = []
    logits_per_step = []
    robot_key = "start"
    robot_world = start_np
    cost = 0.0

    for _ in range(n):
        vec = static_logits[robot_key]
        logits_per_step.append(vec.tolist())  # outgoing-from-current-node scores
        masked = np.where(visited, -np.inf, vec)
        pick = int(np.argmax(masked))
        bin_pos = bin_np[type_np[pick]]
        cost += float(np.linalg.norm(robot_world - obj_np[pick]))
        cost += float(np.linalg.norm(obj_np[pick] - bin_pos))
        route.append(pick)
        visited[pick] = True
        robot_world = bin_pos
        robot_key = int(type_np[pick])

    return cost, route, logits_per_step
