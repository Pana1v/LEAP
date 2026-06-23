"""LEAP k-sweep: map the speed/quality knee of arc pruning.

For each N, do ONE GNN rollout per scenario (rollout cost is k-independent),
then solve the pruned circuit at several k values, reusing the same logits.
Report, per k: mean gap-to-optimum (vs cached CP-SAT optima), mean pruned-solve
time, arcs kept, and LEAP total = rollout + solve.

Overlay the measured GLS anytime curve so each k-point is flagged WIN/LOSS at
iso-time: LEAP wins iff its (total_ms, gap) lies below GLS's gap-at-that-time.

ortools + torch + torch_geometric required. Run from src/ with the torch venv:
    /home/pan-navigator/binning_venv/bin/python bench_leap_k_sweep.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gnn_ilp_circuit import (
    solve_circuit_pruned, select_model_by_count, load_scenarios,
    prepare_scenario, gnn_rollout_with_logits,
)
from gnn_gui import load_model
from gnn_train import split_dataset
from static_rollout import static_rollout_with_logits

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CACHED = REPO / "experiments" / "ilp_timing_circuit_v8.json"
OUT = REPO / "experiments" / "leap_k_sweep_v9.json"

VAL_SPLIT, VAL_SEED, N_WARMUP = 0.1, 42, 10

# Measured GLS gap-to-optimum vs time budget (CRITICAL_findings doc, GLS best-of-3).
GLS_ANYTIME = {
    100: [(100, 0.025), (200, 0.020), (400, 0.015), (800, 0.011)],
    200: [(100, 0.131), (200, 0.041), (400, 0.0075), (800, 0.0038)],
}


def gls_gap_at(n, t_ms):
    """Piecewise log-log interpolation of the GLS anytime curve, clamped."""
    pts = GLS_ANYTIME.get(n)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    if t_ms <= xs[0]:
        return pts[0][1]
    if t_ms >= xs[-1]:
        return pts[-1][1]
    lt = np.log(t_ms)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= t_ms <= x1:
            f = (lt - np.log(x0)) / (np.log(x1) - np.log(x0))
            return float(np.exp(np.log(y0) + f * (np.log(y1) - np.log(y0))))
    return pts[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 8, 10, 12, 15, 20, 25])
    ap.add_argument("--scorer", choices=["auto", "static"], default="auto",
                    help="auto = autoregressive GNN rollout; static = one-shot 5-pass scorer")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rollout_fn = static_rollout_with_logits if args.scorer == "static" else gnn_rollout_with_logits
    device = torch.device(args.device)
    cached = json.loads(CACHED.read_text())["per_n"]
    print(f"device={device}  ks={args.ks}  scorer={args.scorer}\n")

    out_path = OUT.with_name(f"leap_k_sweep_v9_{args.scorer}.json")
    results = {"config": {"ks": args.ks, "device": str(device), "scorer": args.scorer,
                          "val_split": VAL_SPLIT, "val_seed": VAL_SEED}, "per_n": {}}

    for n in args.ns:
        cn = cached.get(str(n))
        if not cn or not cn.get("unpruned"):
            print(f"[skip] N={n}: no cached optima"); continue
        opt_costs = cn["unpruned"]["costs"]
        n_scen = len(opt_costs)

        scenarios = load_scenarios(DATA / f"dataset_{n}_objects.json")
        _, val = split_dataset(scenarios, VAL_SPLIT, VAL_SEED)
        subset = val[:n_scen]
        mp = select_model_by_count(n)
        if not mp.exists():
            print(f"[skip] N={n}: no model at {mp}"); continue
        model = load_model(device, str(mp))

        # warmup at the largest k
        kmax = max(args.ks)
        for s in subset[:N_WARMUP]:
            st = prepare_scenario(s, device)
            _, seq, lg = rollout_fn(model, st, device)
            solve_circuit_pruned(s, seq, lg, k_neighbors=kmax, max_objects=len(s["objects"]) + 2)

        # one rollout per scenario, reused across all k
        roll_ms = []
        per_scen = []  # (scenario, seq, logits, opt)
        for s, o in zip(subset, opt_costs):
            st = prepare_scenario(s, device)
            t0 = time.time()
            _, seq, lg = rollout_fn(model, st, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            roll_ms.append((time.time() - t0) * 1000)
            per_scen.append((s, seq, lg, o))
        rollout_mean = float(np.mean(roll_ms))

        print(f"=== N={n} ({n_scen} scen.) | rollout {rollout_mean:.1f} ms (k-independent) ===")
        print(f"  {'k':>3}{'gap%':>9}{'worst%':>9}{'solve_ms':>10}{'total_ms':>10}"
              f"{'arcs':>7}{'GLS@t%':>9}  verdict")

        sweep = []
        for k in args.ks:
            gaps, solves, arcs = [], [], []
            for (s, seq, lg, o) in per_scen:
                cost, solve_s, n_arcs = solve_circuit_pruned(
                    s, seq, lg, k_neighbors=k, max_objects=len(s["objects"]) + 2)
                gaps.append((cost - o) / o * 100)
                solves.append(solve_s * 1000)
                arcs.append(n_arcs)
            mean_gap = float(np.mean(gaps))
            worst_gap = float(np.max(gaps))
            solve_mean = float(np.mean(solves))
            total = rollout_mean + solve_mean
            gls_at_total = gls_gap_at(n, total)
            win = gls_at_total is not None and mean_gap < gls_at_total
            verdict = "WIN " if win else "loss"
            print(f"  {k:>3}{mean_gap:>9.4f}{worst_gap:>9.4f}{solve_mean:>10.1f}"
                  f"{total:>10.1f}{np.mean(arcs):>7.0f}{gls_at_total:>9.4f}  {verdict}")
            sweep.append({"k": k, "mean_gap_pct": mean_gap, "worst_gap_pct": worst_gap,
                          "solve_ms": solve_mean, "total_ms": total,
                          "arcs": float(np.mean(arcs)), "gls_gap_at_total": gls_at_total,
                          "iso_time_win": bool(win)})

        wins = [s for s in sweep if s["iso_time_win"]]
        if wins:
            best = min(wins, key=lambda s: s["total_ms"])
            print(f"  -> iso-time WINS at k in {[w['k'] for w in wins]}; "
                  f"fastest win: k={best['k']} ({best['total_ms']:.0f} ms, gap {best['mean_gap_pct']:.4f}%)")
        else:
            print(f"  -> no iso-time win in this k range")
        results["per_n"][str(n)] = {"rollout_ms": rollout_mean, "sweep": sweep,
                                    "iso_time_wins": [w["k"] for w in wins]}
        print()
        out_path.write_text(json.dumps(results, indent=2))

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
