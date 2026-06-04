"""
Training metrics tracker with structured logging and visualization.

Accumulates per-epoch loss, accuracy, learning rate, and timing data.
Generates publication-quality multi-panel plots and exports to CSV/JSON
for external analysis. Designed as a drop-in upgrade from DebugTracker.

Typical usage:
    from src.training.metrics_tracker import MetricsTracker

    tracker = MetricsTracker()

    for epoch in range(n_epochs):
        train_loss, train_acc = train_one_epoch(...)
        val_loss, val_acc = validate(...)
        tracker.update(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=optimizer.param_groups[0]['lr'],
            epoch_time=elapsed,
        )

    tracker.save_plots(Path("results/training_curves.png"))
    tracker.save_csv(Path("results/history.csv"))
    tracker.print_info()
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Per-epoch training metrics accumulator with visualization and export.

    Tracks training and validation loss/accuracy, learning rate, and
    epoch wall-clock time. Generates a four-panel summary plot and
    exports structured data for external analysis.

    Attributes:
        epochs: List of epoch numbers.
        train_loss: Training loss per epoch.
        train_acc: Training accuracy per epoch.
        val_loss: Validation loss per epoch.
        val_acc: Validation accuracy per epoch.
        learning_rates: Learning rate per epoch.
        epoch_times: Wall-clock time in seconds per epoch.
    """

    def __init__(self) -> None:
        """Initialize an empty tracker."""
        self.epochs: List[int] = []
        self.train_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_loss: List[float] = []
        self.val_acc: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []

        logger.debug("MetricsTracker initialized")

    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float,
    ) -> None:
        """
        Record metrics for one epoch.

        Args:
            epoch: Epoch number (1-based).
            train_loss: Average training loss.
            train_acc: Training accuracy (0.0 to 1.0).
            val_loss: Average validation loss.
            val_acc: Validation accuracy (0.0 to 1.0).
            lr: Current learning rate.
            epoch_time: Wall-clock time for this epoch in seconds.
        """
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

        logger.info(
            "Epoch %d — train: loss=%.4f acc=%.4f | "
            "val: loss=%.4f acc=%.4f | lr=%.2e | time=%.1fs",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            lr,
            epoch_time,
        )

    def get_best_epoch(self, metric: str = "val_acc") -> int:
        """
        Return the epoch number with the best value for a given metric.

        Args:
            metric: One of 'val_acc', 'val_loss', 'train_acc', 'train_loss'.

        Returns:
            1-based epoch number.

        Raises:
            ValueError: If no metrics have been recorded or metric is unknown.
        """
        if not self.epochs:
            raise ValueError("No metrics recorded yet.")

        metric_map = {
            "val_acc": self.val_acc,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "train_loss": self.train_loss,
        }

        if metric not in metric_map:
            raise ValueError(
                f"Unknown metric: '{metric}'. "
                f"Choose from: {list(metric_map.keys())}"
            )

        values = metric_map[metric]
        if metric.endswith("_loss"):
            idx = int(np.argmin(values))
        else:
            idx = int(np.argmax(values))

        return self.epochs[idx]

    def get_best_metrics(self) -> Dict[str, float]:
        """
        Return the metrics from the best validation accuracy epoch.

        Returns:
            Dictionary with keys: epoch, train_loss, train_acc,
            val_loss, val_acc, lr, epoch_time.
        """
        best_epoch = self.get_best_epoch("val_acc")
        idx = self.epochs.index(best_epoch)

        return {
            "epoch": best_epoch,
            "train_loss": self.train_loss[idx],
            "train_acc": self.train_acc[idx],
            "val_loss": self.val_loss[idx],
            "val_acc": self.val_acc[idx],
            "lr": self.learning_rates[idx],
            "epoch_time": self.epoch_times[idx],
        }

    def get_final_metrics(self) -> Dict[str, float]:
        """
        Return the metrics from the last recorded epoch.

        Returns:
            Dictionary with keys matching get_best_metrics().

        Raises:
            ValueError: If no metrics have been recorded.
        """
        if not self.epochs:
            raise ValueError("No metrics recorded yet.")

        return {
            "epoch": self.epochs[-1],
            "train_loss": self.train_loss[-1],
            "train_acc": self.train_acc[-1],
            "val_loss": self.val_loss[-1],
            "val_acc": self.val_acc[-1],
            "lr": self.learning_rates[-1],
            "epoch_time": self.epoch_times[-1],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all tracked metrics as a pandas DataFrame.

        Returns:
            DataFrame with columns: epoch, train_loss, train_acc,
            val_loss, val_acc, lr, epoch_time_sec.
        """
        return pd.DataFrame({
            "epoch": self.epochs,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "lr": self.learning_rates,
            "epoch_time_sec": self.epoch_times,
        })

    def save_csv(self, save_path: Path) -> None:
        """
        Persist tracked metrics as a CSV file.

        Args:
            save_path: File path for the saved CSV file.
        """
        if not self.epochs:
            logger.warning("No metrics to save — save_csv skipped")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(save_path, index=False)

        logger.info(
            "Metrics CSV saved: %s (%d epochs)",
            save_path,
            len(self.epochs),
        )

    def save_json(self, save_path: Path) -> None:
        """
        Persist tracked metrics as a JSON file.

        Args:
            save_path: File path for the saved JSON file.
        """
        if not self.epochs:
            logger.warning("No metrics to save — save_json skipped")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
            "epoch_times_sec": self.epoch_times,
            "total_time_min": sum(self.epoch_times) / 60,
            "best_epoch": self.get_best_epoch("val_acc"),
            "best_metrics": self.get_best_metrics(),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            "Metrics JSON saved: %s (%d epochs)",
            save_path,
            len(self.epochs),
        )

    def save_plots(self, save_path: Path) -> None:
        """
        Generate and save a four-panel training summary plot.

        Panels:
            1. Loss (train + val)
            2. Accuracy (train + val)
            3. Learning rate over epochs
            4. Epoch wall-clock time

        Args:
            save_path: File path for the saved PNG image.
        """
        if not self.epochs:
            logger.warning("No metrics to plot — save_plots skipped")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        best_epoch = self.get_best_epoch("val_acc")
        epochs_array = np.array(self.epochs)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(epochs_array, self.train_loss, "b-", linewidth=1.5, alpha=0.7, label="Train")
        ax.plot(epochs_array, self.val_loss, "r-", linewidth=1.5, label="Val")
        ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5,
                   label=f"Best (epoch {best_epoch})")
        ax.set_title("Loss", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(epochs_array, self.train_acc, "b-", linewidth=1.5, alpha=0.7, label="Train")
        ax.plot(epochs_array, self.val_acc, "r-", linewidth=1.5, label="Val")
        ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5,
                   label=f"Best (epoch {best_epoch})")
        ax.set_title("Accuracy", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(epochs_array, self.learning_rates, "g-", linewidth=1.5, marker="o", markersize=4)
        ax.set_title("Learning Rate", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.bar(epochs_array, self.epoch_times, color="steelblue", alpha=0.7)
        avg_time = np.mean(self.epoch_times)
        ax.axhline(y=avg_time, color="red", linestyle="--", alpha=0.7,
                   label=f"Mean: {avg_time:.1f}s")
        ax.set_title("Epoch Wall Time", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time (seconds)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        best = self.get_best_metrics()
        fig.suptitle(
            f"Training Summary — Best val_acc: {best['val_acc']:.4f} "
            f"(epoch {best_epoch}), Total: {sum(self.epoch_times)/60:.1f} min",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(
            "Training plots saved: %s (%d epochs)",
            save_path,
            len(self.epochs),
        )

    def print_info(self) -> None:
        """
        Print a summary of tracked metrics.

        This is a manual debugging/exploration utility. Shows best epoch,
        final epoch, and total training time.
        """
        if not self.epochs:
            print("No metrics recorded yet.")
            return

        best = self.get_best_metrics()
        final = self.get_final_metrics()
        total_min = sum(self.epoch_times) / 60

        print("=" * 60)
        print("TRAINING METRICS SUMMARY")
        print("=" * 60)
        print(f"Epochs completed:     {len(self.epochs)}")
        print(f"Total time:           {total_min:.1f} min "
              f"({np.mean(self.epoch_times):.1f} sec/epoch)")
        print(f"\nBest epoch ({best['epoch']}):")
        print(f"  Train loss:         {best['train_loss']:.4f}")
        print(f"  Train acc:          {best['train_acc']:.4f}")
        print(f"  Val loss:           {best['val_loss']:.4f}")
        print(f"  Val acc:            {best['val_acc']:.4f}")
        print(f"\nFinal epoch ({final['epoch']}):")
        print(f"  Train loss:         {final['train_loss']:.4f}")
        print(f"  Train acc:          {final['train_acc']:.4f}")
        print(f"  Val loss:           {final['val_loss']:.4f}")
        print(f"  Val acc:            {final['val_acc']:.4f}")
        print(f"  Learning rate:      {final['lr']:.2e}")
        print("=" * 60)