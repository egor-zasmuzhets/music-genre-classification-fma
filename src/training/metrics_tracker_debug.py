"""
Lightweight metrics tracker for training loop debugging and visualization.

Stores per-epoch loss and accuracy values, generates simple side-by-side
plots, and persists raw data as JSON for later analysis.

Typical usage:
    from src.training.metrics_tracker_debug import DebugTracker

    tracker = DebugTracker()

    for epoch in range(n_epochs):
        loss, acc = train_one_epoch(...)
        tracker.update(loss, acc)

    tracker.save_plots(Path("results/debug_training.png"))
    tracker.save_data(Path("results/debug_metrics.json"))
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class DebugTracker:
    """
    Minimal per-epoch metric tracker for pipeline debugging.

    Accumulates scalar loss and accuracy values across epochs
    and provides simple visualization and JSON export.

    Not intended for production use — use a full-featured tracker
    (e.g., TensorBoard, MLflow) for real experiments.

    Attributes:
        losses: List of per-epoch loss values.
        accs: List of per-epoch accuracy values.
    """

    def __init__(self) -> None:
        """Initialize an empty tracker."""
        self.losses: List[float] = []
        self.accs: List[float] = []

        logger.debug("DebugTracker initialized")

    def update(self, loss: float, acc: float) -> None:
        """
        Record metrics for one epoch.

        Args:
            loss: Average training loss for the epoch.
            acc: Training accuracy as a fraction (0.0 to 1.0).
        """
        self.losses.append(loss)
        self.accs.append(acc)

        logger.info(
            "Epoch %d — loss=%.4f, acc=%.4f",
            len(self.losses),
            loss,
            acc,
        )

    def save_plots(self, save_path: Path) -> None:
        """
        Generate and save a side-by-side loss/accuracy plot.

        Args:
            save_path: File path for the saved PNG image.
        """
        if not self.losses:
            logger.warning("No metrics to plot — save_plots skipped")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        epochs = list(range(1, len(self.losses) + 1))

        ax1.plot(epochs, self.losses, "b-", marker="o", markersize=4)
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.accs, "g-", marker="o", markersize=4)
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, alpha=0.3)

        if self.losses:
            ax1.annotate(
                f"{self.losses[-1]:.4f}",
                (epochs[-1], self.losses[-1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
            ax2.annotate(
                f"{self.accs[-1]:.4f}",
                (epochs[-1], self.accs[-1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()

        logger.info(
            "Training plots saved: %s (%d epochs)",
            save_path,
            len(self.losses),
        )

    def save_data(self, save_path: Path) -> None:
        """
        Persist raw metrics as a JSON file.

        Args:
            save_path: File path for the saved JSON file.
        """
        if not self.losses:
            logger.warning("No metrics to save — save_data skipped")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {"loss": self.losses, "accuracy": self.accs},
                f,
                indent=2,
            )

        logger.info(
            "Metrics data saved: %s (%d epochs)",
            save_path,
            len(self.losses),
        )

    def get_final_metrics(self) -> Tuple[float, float]:
        """
        Return the last recorded loss and accuracy.

        Returns:
            Tuple of (loss, accuracy).

        Raises:
            ValueError: If no metrics have been recorded.
        """
        if not self.losses:
            raise ValueError("No metrics recorded yet.")
        return self.losses[-1], self.accs[-1]

    def print_info(self) -> None:
        """
        Print a summary of recorded metrics.

        This is a manual debugging/exploration utility.
        """
        if not self.losses:
            print("No metrics recorded yet.")
            return

        best_acc = max(self.accs)
        best_epoch = self.accs.index(best_acc) + 1

        print("=" * 40)
        print("DEBUG TRACKER SUMMARY")
        print("=" * 40)
        print(f"Epochs recorded: {len(self.losses)}")
        print(f"Final loss:      {self.losses[-1]:.4f}")
        print(f"Final accuracy:  {self.accs[-1]:.4f}")
        print(f"Best accuracy:   {best_acc:.4f} (epoch {best_epoch})")
        print("=" * 40)


def check_dataloader(
    loader: DataLoader,
    name: str = "data",
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Validate that a DataLoader produces correctly shaped batches.

    Retrieves one batch and logs its shape, value range, and unique labels.
    Useful for pipeline debugging before starting full training.

    Args:
        loader: PyTorch DataLoader to inspect.
        name: Human-readable name for log attribution.

    Returns:
        Tuple of (features, labels) from the first batch, or None on failure.

    Raises:
        RuntimeError: If the DataLoader is empty or produces malformed batches.
    """
    try:
        batch = next(iter(loader))
        x, y = batch

        logger.info(
            "[%s] DataLoader OK — X: %s, y: %s, "
            "value range: [%.3f, %.3f], unique classes: %s",
            name,
            tuple(x.shape),
            tuple(y.shape),
            x.min().item(),
            x.max().item(),
            y.unique().tolist(),
        )

        return batch

    except StopIteration:
        logger.error("[%s] DataLoader is empty — no batches available", name)
        raise RuntimeError(f"DataLoader '{name}' is empty")

    except Exception:
        logger.error("[%s] DataLoader check failed", name, exc_info=True)
        raise