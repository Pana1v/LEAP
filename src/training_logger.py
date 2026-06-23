import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


def extract_dataset_info(dataset_path) -> str:
    return Path(dataset_path).stem


class TrainingLogger:
    def __init__(
        self,
        dataset_name: str,
        model_type: str,
        base_log_dir: str = "logs",
        hyperparams: Optional[Dict] = None,
    ):
        self.dataset_name = dataset_name
        self.model_type = model_type
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(base_log_dir) / f"{model_type}_{dataset_name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.hyperparams = hyperparams or {}
        self.epochs: List[Dict] = []
        self.stage_transitions: List[Dict] = []
        self.final_metrics: Optional[Dict] = None

        if hyperparams:
            with open(self.log_dir / "hyperparams.json", "w") as f:
                json.dump(hyperparams, f, indent=2)

        self._csv_path = self.log_dir / "training_log.csv"
        self._csv_fields = [
            "epoch", "stage", "train_loss", "learning_rate", "grad_norm",
            "val_mean_cost", "val_mean_gap_vs_greedy", "val_gap_std", "val_win_rate",
            "val_mean_gap_vs_ilp", "val_mean_cost_random",
        ]
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
            writer.writeheader()

    def log_stage_transition(self, from_stage: int, to_stage: int, epoch: int):
        self.stage_transitions.append(
            {"from_stage": from_stage, "to_stage": to_stage, "epoch": epoch}
        )

    def log_epoch(
        self,
        epoch: int,
        stage: int,
        train_loss: float,
        val_metrics: Dict,
        learning_rate: float,
        grad_norm: float = 0.0,
    ):
        row = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        self.epochs.append(row)

        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
            writer.writerow(row)

    def log_final_evaluation(self, metrics: Dict, beam_search: bool = False):
        self.final_metrics = {"metrics": metrics, "beam_search": beam_search}

    def generate_plots(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            if not self.epochs:
                return

            epochs = [e["epoch"] for e in self.epochs]
            losses = [e["train_loss"] for e in self.epochs]
            gaps = [e.get("val_mean_gap_vs_greedy", 0) for e in self.epochs]
            gap_stds = [e.get("val_gap_std", 0) for e in self.epochs]
            ilp_gaps = [e.get("val_mean_gap_vs_ilp") for e in self.epochs]
            lrs = [e.get("learning_rate", 0) for e in self.epochs]
            grad_norms = [e.get("grad_norm", 0) for e in self.epochs]
            costs = [e.get("val_mean_cost", 0) for e in self.epochs]
            win_rates = [e.get("val_win_rate", 0) for e in self.epochs]

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # [0, 0] Training Loss
            axes[0, 0].plot(epochs, losses, color='C0', linewidth=2)
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Train Loss")
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].grid(True, alpha=0.3)

            # [0, 1] Val Gap vs Greedy with std band
            axes[0, 1].plot(epochs, gaps, color='C1', linewidth=2, label='Mean Gap')
            gap_upper = [g + s for g, s in zip(gaps, gap_stds)]
            gap_lower = [g - s for g, s in zip(gaps, gap_stds)]
            axes[0, 1].fill_between(epochs, gap_lower, gap_upper, alpha=0.2, color='C1', label='±1σ')
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Gap vs Greedy (%)")
            axes[0, 1].set_title("Validation Gap vs Greedy")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # [0, 2] Val Gap vs ILP
            ilp_gaps_valid = [g for g in ilp_gaps if g is not None]
            if ilp_gaps_valid:
                ilp_epochs = [e for e, g in zip(epochs, ilp_gaps) if g is not None]
                axes[0, 2].plot(ilp_epochs, ilp_gaps_valid, color='C2', linewidth=2)
                axes[0, 2].set_xlabel("Epoch")
                axes[0, 2].set_ylabel("Gap vs ILP (%)")
                axes[0, 2].set_title("Validation Gap vs ILP")
            else:
                axes[0, 2].text(0.5, 0.5, "No ILP data", ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title("Validation Gap vs ILP")
            axes[0, 2].grid(True, alpha=0.3)

            # [1, 0] Learning Rate (log scale)
            axes[1, 0].semilogy(epochs, lrs, color='C3', linewidth=2)
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Learning Rate (log)")
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].grid(True, alpha=0.3, which='both')

            # [1, 1] Gradient Norm
            axes[1, 1].plot(epochs, grad_norms, color='C4', linewidth=2)
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Gradient Norm")
            axes[1, 1].set_title("Gradient Norm")
            axes[1, 1].grid(True, alpha=0.3)

            # [1, 2] Win Rate
            axes[1, 2].plot(epochs, win_rates, color='C5', linewidth=2, marker='o', markersize=4)
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("Win Rate (%)")
            axes[1, 2].set_title("Win Rate vs Greedy")
            axes[1, 2].set_ylim([0, 100])
            axes[1, 2].grid(True, alpha=0.3)

            # Mark stage transitions on all subplots
            for ax in axes.flat:
                for t in self.stage_transitions:
                    ax.axvline(x=t["epoch"], color="r", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(self.log_dir / "training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            pass

    def save(self):
        summary = {
            "dataset_name": self.dataset_name,
            "model_type": self.model_type,
            "hyperparams": self.hyperparams,
            "stage_transitions": self.stage_transitions,
            "final_metrics": self.final_metrics,
            "num_epochs": len(self.epochs),
        }
        with open(self.log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
