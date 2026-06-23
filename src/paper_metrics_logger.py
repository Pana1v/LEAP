import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch


def get_model_name(object_count: int, model_type: str, suffix: str = "best") -> str:
    return f"{model_type}_{object_count}obj_{suffix}.pt"


class PaperMetricsLogger:
    def __init__(
        self,
        dataset_path,
        model_type: str,
        experiment_name: str,
        base_dir: str = "experiments",
        hyperparams: Optional[Dict] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.model_type = model_type
        self.experiment_name = experiment_name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.hyperparams = hyperparams or {}
        self.epochs: List[Dict] = []
        self.final_metrics: Optional[Dict] = None
        self.inference_stats: Optional[Dict] = None

        if hyperparams:
            with open(self.out_dir / "hyperparams.json", "w") as f:
                json.dump(hyperparams, f, indent=2, default=str)

        self._csv_path = self.out_dir / "epoch_metrics.csv"
        self._csv_fields = [
            "epoch", "stage", "train_loss", "learning_rate", "grad_norm", "is_best",
            "val_mean_cost", "val_mean_gap_vs_greedy", "val_gap_std", "val_win_rate",
            "val_mean_gap_vs_ilp", "val_mean_cost_random",
        ]
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
            writer.writeheader()

    def log_epoch(
        self,
        epoch: int,
        stage: int,
        train_loss: float,
        val_metrics: Dict,
        learning_rate: float,
        grad_norm: float = 0.0,
        is_best: bool = False,
    ):
        row = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "is_best": is_best,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        self.epochs.append(row)

        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
            writer.writerow(row)

    def log_model_info(self, model: torch.nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info = {"total_params": total_params, "trainable_params": trainable}
        with open(self.out_dir / "model_info.json", "w") as f:
            json.dump(info, f, indent=2)

    def benchmark_inference(
        self,
        model: torch.nn.Module,
        inference_fn: Callable,
        num_samples: int = 100,
        warmup_samples: int = 10,
        device: torch.device = None,
    ) -> Dict:
        model.eval()
        with torch.no_grad():
            for _ in range(warmup_samples):
                inference_fn()

            times = []
            for _ in range(num_samples):
                if device and device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                inference_fn()
                if device and device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        stats = {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "median_ms": float(np.median(times) * 1000),
            "p95_ms": float(np.percentile(times, 95) * 1000),
            "num_samples": num_samples,
        }
        self.inference_stats = stats
        with open(self.out_dir / "inference_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        model.train()
        return stats

    def save_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        suffix: str = "best",
        **extra,
    ):
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            **extra,
        }
        path = self.out_dir / f"model_{suffix}.pt"
        torch.save(checkpoint, path)

    def log_final_evaluation(
        self,
        metrics: Dict,
        beam_search: bool = False,
        inference_stats: Optional[Dict] = None,
    ):
        self.final_metrics = {
            "metrics": metrics,
            "beam_search": beam_search,
            "inference_stats": inference_stats,
        }

    def finalize(self):
        summary = {
            "experiment_name": self.experiment_name,
            "model_type": self.model_type,
            "dataset": str(self.dataset_path),
            "hyperparams": self.hyperparams,
            "final_metrics": self.final_metrics,
            "inference_stats": self.inference_stats,
            "num_epochs": len(self.epochs),
        }
        with open(self.out_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Paper metrics saved to: {self.out_dir}")
