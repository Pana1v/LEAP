"""
Inference time benchmarking across problem sizes.
Compares GNN vs Greedy (NC) vs ILP timing.

Run from src/:  python3 benchmark_inference.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnn_train import (
    GNNPolicy,
    FEATURE_DIM,
    DEFAULT_DROPOUT,
    WORKSPACE_SIZE,
    NUM_BINS,
    load_scenarios,
    split_dataset,
    prepare_scenario,
    rollout_model,
    _build_step_graph,
)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "experiments"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "paper"

# Publication-quality plot settings
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "serif",
})

CONFIGS = [
    (10,  MODEL_DIR / "gnn_10obj_best.pt",    DATA_DIR / "dataset_10_objects.json"),
    (20,  None,                                 DATA_DIR / "dataset_20_objects.json"),
    (40,  MODEL_DIR / "gnn_final_40obj.pt",    DATA_DIR / "dataset_40_objects.json"),
    (200, MODEL_DIR / "gnn_final_200obj.pt",   DATA_DIR / "dataset_200_objects.json"),
]

NUM_SAMPLES = 100


def load_model(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
        hd = int(blob.get("hidden_dim", 128))
        heads = int(blob.get("heads", 4))
        dp = float(blob.get("dropout", DEFAULT_DROPOUT))
    else:
        sd = blob
        hd = sd["convs.0.bias"].shape[0] if "convs.0.bias" in sd else 128
        heads = sd["convs.0.att_src"].shape[1] if "convs.0.att_src" in sd else 4
        dp = DEFAULT_DROPOUT
    model = GNNPolicy(FEATURE_DIM, hd, heads, dp).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def time_gnn(model, scenarios, device):
    """Time GNN rollouts."""
    times = []
    # Warmup: ≥10 dummy passes to stabilize caches/JIT/allocators
    with torch.no_grad():
        if scenarios:
            for _ in range(10):
                rollout_model(model, scenarios[0], device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    cuda = torch.cuda.is_available()
    with torch.no_grad():
        for s in scenarios:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            rollout_model(model, s, device)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def time_greedy_nc(scenarios_raw):
    """Time greedy nearest-cycle computation."""
    times = []
    for s in scenarios_raw:
        t0 = time.perf_counter()
        objects = s["objects"]
        types = s["types"]
        bins = s["bins"]
        start = s["start"]
        n = len(objects)
        remaining = set(range(n))
        robot = list(start)
        cost = 0.0
        for _ in range(n):
            best_idx, best_cost = -1, float("inf")
            for i in remaining:
                ox, oy = objects[i]
                bx, by = bins[types[i]]
                c = ((robot[0]-ox)**2 + (robot[1]-oy)**2)**0.5 + ((ox-bx)**2 + (oy-by)**2)**0.5
                if c < best_cost:
                    best_cost = c
                    best_idx = i
            remaining.remove(best_idx)
            bx, by = bins[types[best_idx]]
            robot = [bx, by]
            cost += best_cost
        times.append(time.perf_counter() - t0)
    return times


def time_ilp(scenarios_raw, timeout=60.0):
    """Time exact ILP solving via CP-SAT `AddCircuit()` (replaces legacy CBC+MTZ)."""
    try:
        from gnn_ilp_circuit import solve_circuit_cold
    except ImportError:
        return None

    times = []
    for s in scenarios_raw:
        n = len(s["objects"])
        if n > 60:
            times.append(float("inf"))
            continue
        t0 = time.perf_counter()
        try:
            _cost, _elapsed = solve_circuit_cold(s, max_objects=max(n + 2, 60))
        except RuntimeError:
            times.append(float("inf"))
            continue
        times.append(time.perf_counter() - t0)
    return times


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ilp", action="store_true", help="Skip ILP timing (reuse existing)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    existing = {}
    out_path_check = OUT_DIR / "timing_results.json"
    if args.no_ilp and out_path_check.exists():
        with open(out_path_check) as f:
            existing = json.load(f)

    results = {}

    for n, model_path, dataset_path in CONFIGS:
        if not dataset_path.exists():
            print(f"Skip N={n}: dataset not found")
            continue

        print(f"\n{'='*50}")
        print(f"Benchmarking N={n}")
        print(f"{'='*50}")

        scenarios_raw = load_scenarios(dataset_path)
        _, val_raw = split_dataset(scenarios_raw, 0.1, 0)
        val_raw = val_raw[:NUM_SAMPLES]

        entry = {"n": n}

        # GNN timing
        if model_path and model_path.exists():
            val_prep = [prepare_scenario(s, device) for s in val_raw]
            model = load_model(model_path, device)
            gnn_times = time_gnn(model, val_prep, device)
            gnn_ms = [t * 1000 for t in gnn_times]
            entry["gnn"] = {
                "mean_ms": float(np.mean(gnn_ms)),
                "std_ms": float(np.std(gnn_ms)),
                "median_ms": float(np.median(gnn_ms)),
                "p95_ms": float(np.percentile(gnn_ms, 95)),
            }
            print(f"  GNN:    mean={entry['gnn']['mean_ms']:.1f}ms  "
                  f"std={entry['gnn']['std_ms']:.1f}ms  p95={entry['gnn']['p95_ms']:.1f}ms")
        else:
            entry["gnn"] = None
            print(f"  GNN:    no model available")

        # Greedy NC timing
        greedy_times = time_greedy_nc(val_raw)
        greedy_ms = [t * 1000 for t in greedy_times]
        entry["greedy_nc"] = {
            "mean_ms": float(np.mean(greedy_ms)),
            "std_ms": float(np.std(greedy_ms)),
            "median_ms": float(np.median(greedy_ms)),
            "p95_ms": float(np.percentile(greedy_ms, 95)),
        }
        print(f"  Greedy: mean={entry['greedy_nc']['mean_ms']:.2f}ms  "
              f"std={entry['greedy_nc']['std_ms']:.2f}ms")

        # ILP timing (only for N <= 40)
        if args.no_ilp:
            prev = existing.get(str(n), {}).get("ilp")
            entry["ilp"] = prev if prev is not None else {"note": "skipped"}
            print(f"  ILP:    reused from existing results")
        elif n <= 40:
            ilp_samples = val_raw[:10]  # ILP is slow, cap at 10 samples
            ilp_times = time_ilp(ilp_samples, timeout=30.0)
            if ilp_times:
                ilp_ms = [t * 1000 for t in ilp_times if t < 60.0]
                if ilp_ms:
                    entry["ilp"] = {
                        "mean_ms": float(np.mean(ilp_ms)),
                        "std_ms": float(np.std(ilp_ms)),
                        "median_ms": float(np.median(ilp_ms)),
                        "p95_ms": float(np.percentile(ilp_ms, 95)),
                        "n_solved": len(ilp_ms),
                        "n_timeout": len(ilp_times) - len(ilp_ms),
                    }
                    print(f"  ILP:    mean={entry['ilp']['mean_ms']:.1f}ms  "
                          f"({entry['ilp']['n_solved']}/{len(ilp_times)} solved)")
                else:
                    entry["ilp"] = {"note": "all timed out"}
                    print(f"  ILP:    all timed out")
            else:
                entry["ilp"] = None
                print(f"  ILP:    ortools not available")
        else:
            entry["ilp"] = {"note": "intractable"}
            print(f"  ILP:    intractable at N={n}")

        results[str(n)] = entry

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "timing_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Generate plots
    _plot_timing(results)


def _plot_timing(results):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ns = sorted(int(k) for k in results.keys())
    gnn_means, greedy_means, ilp_means = [], [], []
    gnn_ns, greedy_ns, ilp_ns = [], [], []

    for n in ns:
        r = results[str(n)]
        if r.get("gnn") and isinstance(r["gnn"], dict) and "mean_ms" in r["gnn"]:
            gnn_ns.append(n)
            gnn_means.append(r["gnn"]["mean_ms"])
        greedy_ns.append(n)
        greedy_means.append(r["greedy_nc"]["mean_ms"])
        if r.get("ilp") and isinstance(r["ilp"], dict) and "mean_ms" in r["ilp"]:
            ilp_ns.append(n)
            ilp_means.append(r["ilp"]["mean_ms"])

    # --- Fig: Timing scaling (log-scale) ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    if gnn_means:
        ax.plot(gnn_ns, gnn_means, "o-", color="#2980b9", lw=2.5, ms=8, label="GNN (ours)", zorder=5)
    ax.plot(greedy_ns, greedy_means, "s--", color="#e74c3c", lw=2, ms=7, label="Greedy (NC)", zorder=4)
    if ilp_means:
        ax.plot(ilp_ns, ilp_means, "D-.", color="#8e44ad", lw=2, ms=7, label="ILP (CBC)", zorder=4)

    # Mark intractable region — place text after axis limits are set
    intractable_ns = []
    for n in ns:
        r = results[str(n)]
        if r.get("ilp") and isinstance(r["ilp"], dict) and r["ilp"].get("note") == "intractable":
            ax.axvspan(n - 10, n + 10, alpha=0.08, color="red")
            intractable_ns.append(n)

    ax.set_xlabel("Number of Objects ($N$)")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Time Scaling by Method")
    ax.set_yscale("log")
    ax.legend(loc="upper left")

    # Place intractable labels now that axis is scaled
    for n in intractable_ns:
        ax.text(n, ax.get_ylim()[1] * 0.5, "ILP\nintractable",
                ha="center", va="top", fontsize=11, color="#c0392b", style="italic")

    fig.savefig(FIG_DIR / "fig_timing_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_timing_scaling.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig_timing_scaling.png'}")

    # --- Fig: Timing breakdown (stacked bar, GNN only) ---
    if gnn_means:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        x = np.arange(len(gnn_ns))
        ax.bar(x, gnn_means, 0.5, color="#2980b9", alpha=0.85, edgecolor="white", label="GNN Total")

        max_val = max(gnn_means)
        ax.set_ylim(0, max_val * 1.18)
        for i, (n, t) in enumerate(zip(gnn_ns, gnn_means)):
            ax.text(i, t + max_val * 0.03, f"{t:.0f} ms", ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_xlabel("Number of Objects ($N$)")
        ax.set_ylabel("Mean Inference Time (ms)")
        ax.set_title("GNN Inference Time by Problem Size")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$N={n}$" for n in gnn_ns])

        fig.savefig(FIG_DIR / "fig_timing_breakdown.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / "fig_timing_breakdown.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {FIG_DIR / 'fig_timing_breakdown.png'}")

    print("Timing plots done.")


if __name__ == "__main__":
    main()
