"""
Generate publication-quality plots from training logs.

Usage:
    python generate_paper_plots.py --log ../logs/training_log.csv --output ../docs/figures/
    python generate_paper_plots.py --log path/to/training_log.csv --output path/to/output/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_training_log(csv_path: Path) -> pd.DataFrame:
    """Load training log CSV."""
    return pd.read_csv(csv_path)


def load_experiment_summary(summary_path: Path) -> dict:
    """Load experiment summary JSON if it exists."""
    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)
    return {}


def plot_learning_curves(df: pd.DataFrame, output_dir: Path):
    """
    Generate learning curves figure (2-column IEEE format).
    Left: Training loss vs. epoch
    Right: Val gap vs. greedy % with ±σ band
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    fig.suptitle("Learning Curves", fontsize=11, weight='bold', y=1.00)

    epochs = df['epoch'].values
    losses = df['train_loss'].values
    gaps = df['val_mean_gap_vs_greedy'].values
    gap_stds = df['val_gap_std'].fillna(0).values

    # Left: Training loss
    axes[0].plot(epochs, losses, 'C0-', linewidth=1.5, markersize=3)
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Training Loss", fontsize=10)
    axes[0].set_title("Training Loss", fontsize=10, pad=8)
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[0].tick_params(labelsize=9)

    # Right: Val gap with std band
    axes[1].plot(epochs, gaps, 'C1-', linewidth=1.5, markersize=3, label='Mean Gap')
    gap_upper = gaps + gap_stds
    gap_lower = gaps - gap_stds
    axes[1].fill_between(epochs, gap_lower, gap_upper, alpha=0.2, color='C1', label='±1σ')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Gap vs Greedy (%)", fontsize=10)
    axes[1].set_title("Validation Gap", fontsize=10, pad=8)
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[1].tick_params(labelsize=10)

    plt.tight_layout()
    output_path = output_dir / "fig_learning_curves.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_dir / "fig_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {output_path.name} and PNG version")


def plot_ilp_gap(df: pd.DataFrame, output_dir: Path):
    """
    Generate ILP gap figure if ILP data is available.
    """
    ilp_gaps = df['val_mean_gap_vs_ilp'].dropna()
    if len(ilp_gaps) == 0:
        print("⊘ Skipping ILP gap plot (no ILP data available)")
        return

    fig, ax = plt.subplots(figsize=(5, 3))

    epochs_with_ilp = df[df['val_mean_gap_vs_ilp'].notna()]['epoch'].values
    ax.plot(epochs_with_ilp, ilp_gaps.values, 'C2-', linewidth=1.5, markersize=4)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Gap vs ILP (%)", fontsize=10)
    ax.set_title("GNN Performance vs ILP", fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    output_path = output_dir / "fig_ilp_gap.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_dir / "fig_ilp_gap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {output_path.name} and PNG version")


def plot_win_rate(df: pd.DataFrame, output_dir: Path):
    """
    Generate win rate figure (% scenarios beating greedy).
    """
    if 'val_win_rate' not in df.columns:
        print("⊘ Skipping win rate plot (no win rate data available)")
        return

    fig, ax = plt.subplots(figsize=(5, 3))

    epochs = df['epoch'].values
    win_rates = df['val_win_rate'].values

    ax.plot(epochs, win_rates, 'C5-', linewidth=1.5, markersize=4, label='Win Rate')
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Win Rate (%)", fontsize=10)
    ax.set_title("GNN Beat Greedy (%)", fontsize=11, weight='bold')
    ax.set_ylim([0, 100])
    ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    output_path = output_dir / "fig_win_rate.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_dir / "fig_win_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {output_path.name} and PNG version")


def plot_method_comparison(final_metrics: dict, output_dir: Path):
    """
    Generate method comparison bar chart (GNN vs Greedy).
    Requires final_metrics dict from summary.json.
    """
    if 'metrics' not in final_metrics:
        print("⊘ Skipping method comparison (no final metrics in summary)")
        return

    metrics = final_metrics['metrics']
    methods = ['GNN', 'Greedy']
    costs = [
        metrics.get('mean_cost', 0),
        metrics.get('greedy_mean_cost', metrics.get('mean_cost_greedy', 0)),
    ]
    stds = [
        metrics.get('std_cost', 0),
        metrics.get('std_cost_greedy', 0),
    ]

    # Filter out zero costs
    valid_idx = [i for i, c in enumerate(costs) if c > 0]
    if not valid_idx:
        print("⊘ Skipping method comparison (no valid cost data)")
        return

    methods = [methods[i] for i in valid_idx]
    costs = [costs[i] for i in valid_idx]
    stds = [stds[i] for i in valid_idx]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(methods))
    width = 0.6
    colors = ['#2ecc71', '#e74c3c'][:len(methods)]
    bars = ax.bar(x, costs, width, yerr=stds, capsize=5, alpha=0.8,
                   color=colors)

    ax.set_ylabel("Mean Cost", fontsize=10)
    ax.set_title("Method Comparison", fontsize=11, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=9)

    # Add value labels on bars
    for i, (cost, std) in enumerate(zip(costs, stds)):
        ax.text(i, cost + std + max(costs) * 0.02, f'{cost:.1f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "fig_method_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_dir / "fig_method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {output_path.name} and PNG version")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality plots from training logs.")
    parser.add_argument("--log", type=str, required=True, help="Path to training_log.csv")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: same as log directory)")
    parser.add_argument("--summary", type=str, default=None, help="Path to summary.json for final metrics")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = log_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_training_log(log_path)
    print(f"✓ Loaded training log with {len(df)} epochs")

    # Load summary if available
    summary_path = args.summary or log_path.parent / "summary.json"
    summary = load_experiment_summary(summary_path)

    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curves(df, output_dir)
    plot_ilp_gap(df, output_dir)
    plot_win_rate(df, output_dir)
    if summary:
        plot_method_comparison(summary, output_dir)

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("fig_*.pdf")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
