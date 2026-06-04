#!/usr/bin/env python
"""
CNN training script for MFCC-based audio genre classification.

Orchestrates data loading, model creation, training with early stopping,
and comprehensive evaluation with PyTorchModelAnalyzer. All artifacts are
saved into a timestamped experiment directory for reproducibility.

Uses balanced sampling instead of class weights for imbalanced data.

Usage:
    python -m src.training.train_cnn --subset medium --epochs 50
    python -m src.training.train_cnn --subset small --epochs 20 --augment

    from src.training.train_cnn import CNNTrainer, build_model

    model = build_model(n_mfcc=40, n_channels=3, n_classes=16)
    trainer = CNNTrainer(model, "cuda", config)
    trainer.fit(train_loader, val_loader)
    results = trainer.evaluate(test_loader, genre_names)
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.torch_dataset import create_mfcc_dataloaders
from src.models.cnn_audio import AudioCNN
from src.training.metrics_tracker import MetricsTracker
from src.training.pytorch_analyzer import PyTorchModelAnalyzer
from src.utils.config import paths
from src.utils.logging_utils import setup_logging

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_PLOT = True
    matplotlib.use('Agg')
except ImportError:
    HAS_PLOT = False

logger = logging.getLogger(__name__)


def build_model(
    n_mfcc: int = 40,
    n_channels: int = 3,
    n_classes: int = 16,
    dropout: float = 0.3,
    fc_dropout: float = 0.2,
    target_frames: int = 430,
) -> AudioCNN:
    """
    Create an AudioCNN model with the specified configuration.

    Input shape for the model: (batch, n_channels, n_mfcc, time_frames)

    Args:
        n_mfcc: Number of MFCC coefficient bands (default: 40).
        n_channels: Number of input channels (default: 3 with deltas).
        n_classes: Number of genre classes.
        dropout: Dropout rate after convolutional stem (default: 0.3).
        fc_dropout: Dropout rate between FC layers (default: 0.2).
        target_frames: Expected time frames in input (default: 430 ≈ 10s).

    Returns:
        Initialized AudioCNN instance.
    """
    return AudioCNN(
        n_mfcc=n_mfcc,
        n_channels=n_channels,
        n_classes=n_classes,
        dropout=dropout,
        fc_dropout=fc_dropout,
        target_frames=target_frames,
    )


class CNNTrainer:
    """
    Full-featured training loop for CNN models.

    Handles per-epoch training and validation, early stopping,
    learning rate scheduling, intermediate checkpointing, and
    final model persistence with experiment summaries.

    Uses balanced sampling (via WeightedRandomSampler in DataLoader)
    instead of class weights in the loss function.

    Attributes:
        model: The PyTorch model to train.
        device: Torch device string.
        config: Training configuration dictionary.
        experiment_name: Unique name for this training run.
        exp_dir: Path to the experiment output directory.
        best_val_acc: Best validation accuracy achieved.
        best_epoch: Epoch number of the best validation accuracy.
        tracker: MetricsTracker instance accumulating per-epoch data.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        config: Dict[str, Any],
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train.
            device: Torch device string ('cpu', 'cuda', 'cuda:0').
            config: Training configuration dictionary.
            experiment_name: Custom experiment directory name.
        """
        self.model = model
        self.device = device
        self.config = config

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = (
                f"{timestamp}_"
                f"{config.get('subset', 'unknown')}_"
                f"{config.get('min_samples', 0)}_"
                f"{config.get('n_mfcc', 40)}mfcc_"
                f"{config.get('target_frames', 430)}frames_"
                f"{config.get('n_channels', 3)}ch"
            )

        self.experiment_name = experiment_name
        self.exp_dir = paths.cnn.results_dir / experiment_name
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.plots_dir = self.exp_dir / "plots"
        self.metrics_dir = self.exp_dir / "metrics"
        self.analysis_dir = self.exp_dir / "analysis"

        for d in [self.checkpoints_dir, self.plots_dir, self.metrics_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.tracker = MetricsTracker()
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self._optimizer: Optional[optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None
        self._scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None
        self._early_stop_counter = 0

        logger.info("CNNTrainer initialized — experiment: %s", experiment_name)
        logger.info("Experiment directory: %s", self.exp_dir)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure input is 4D (B, C, H, W) for Conv2d.

        Args:
            x: Input tensor from DataLoader, 3D or 4D.

        Returns:
            4D tensor (B, C, H, W).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run a single training epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = self._prepare_input(x).to(self.device)
            y = y.to(self.device)

            self._optimizer.zero_grad()
            out = self.model(x)
            loss = self._criterion(out, y)
            loss.backward()

            grad_clip = self.config.get("grad_clip")
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self._optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        n_batches = len(loader)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on validation set. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = self._prepare_input(x).to(self.device)
            y = y.to(self.device)

            out = self.model(x)
            loss = self._criterion(out, y)

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        n_batches = len(loader)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _save_checkpoint(self, filename: str) -> Path:
        """Save model state dict and current config."""
        path = self.checkpoints_dir / filename
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "model_architecture": {
                "n_mfcc": self.config.get("n_mfcc", 40),
                "n_channels": self.config.get("n_channels", 3),
                "n_classes": len(self.config.get("genre_names", [])),
                "dropout": self.config.get("dropout", 0.3),
                "fc_dropout": self.config.get("fc_dropout", 0.2),
                "target_frames": self.config.get("target_frames", 430),
            },
        }
        torch.save(checkpoint, path)
        return path

    def _save_intermediate(self, epoch: int) -> None:
        """Save intermediate training artifacts."""
        self._save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        self.tracker.save_plots(self.plots_dir / "training_curves.png")
        self.tracker.save_csv(self.metrics_dir / "training_history.csv")

        best = self.tracker.get_best_metrics()
        current_val_acc = self.tracker.val_acc[-1] if self.tracker.val_acc else 0.0
        current_lr = self._optimizer.param_groups[0]["lr"] if self._optimizer else 0.0
        logger.info(
            "Epoch %d — checkpoint saved | "
            "best: val_acc=%.4f (epoch %d) | current: val_acc=%.4f | lr=%.2e",
            epoch,
            best["val_acc"],
            best["epoch"],
            current_val_acc,
            current_lr,
        )

    @torch.no_grad()
    def _log_per_class_f1(
        self,
        loader: DataLoader,
        genre_names: List[str],
        class_counts: Dict[int, int],
        rare_threshold: int = 100,
        medium_threshold: int = 500,
    ) -> None:
        """
        Log per-class F1 scores grouped by rarity.

        Args:
            loader: DataLoader to evaluate on (typically validation).
            genre_names: Ordered list of genre names.
            class_counts: Dict mapping class index to sample count.
            rare_threshold: Max samples for 'rare' class.
            medium_threshold: Max samples for 'medium' class.
        """
        from sklearn.metrics import f1_score

        self.model.eval()
        all_preds = []
        all_labels = []

        for x, y in loader:
            x = self._prepare_input(x).to(self.device)
            out = self.model(x)
            all_preds.append(out.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        rare_f1s = []
        medium_f1s = []
        common_f1s = []

        for i, genre in enumerate(genre_names):
            f1 = f1_score(y_true == i, y_pred == i, zero_division=0)
            count = class_counts.get(i, 0)

            if count <= rare_threshold:
                rare_f1s.append((genre, f1, count))
            elif count <= medium_threshold:
                medium_f1s.append((genre, f1, count))
            else:
                common_f1s.append((genre, f1, count))

        rare_mean = np.mean([f for _, f, _ in rare_f1s]) if rare_f1s else 0.0
        medium_mean = np.mean([f for _, f, _ in medium_f1s]) if medium_f1s else 0.0
        common_mean = np.mean([f for _, f, _ in common_f1s]) if common_f1s else 0.0

        all_f1s = [f for _, f, _ in rare_f1s + medium_f1s + common_f1s]
        macro_f1 = np.mean(all_f1s) if all_f1s else 0.0

        logger.info(
            "--- Per-class F1 (Rare ≤%d | Medium ≤%d | Common >%d) ---",
            rare_threshold, medium_threshold, medium_threshold,
        )

        if rare_f1s:
            rare_str = "  ".join(f"{g}:{f:.3f}" for g, f, _ in rare_f1s)
            logger.info("  Rare:   %s", rare_str)
        if medium_f1s:
            medium_str = "  ".join(f"{g}:{f:.3f}" for g, f, _ in medium_f1s)
            logger.info("  Medium: %s", medium_str)

        logger.info(
            "  F1 → Rare: %.3f | Medium: %.3f | Common: %.3f | Macro: %.3f",
            rare_mean, medium_mean, common_mean, macro_f1,
        )

        # Highlight worst rare class
        if rare_f1s:
            worst_genre, worst_f1, worst_count = min(rare_f1s, key=lambda x: x[1])
            logger.info(
                "  ⚠ Worst rare: %s (F1=%.3f, %d samples)",
                worst_genre, worst_f1, worst_count,
            )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_counts: Optional[Dict[int, int]] = None,
        genre_names: Optional[List[str]] = None,
    ) -> MetricsTracker:
        """
        Run the full training loop.

        Uses standard CrossEntropyLoss without class weights —
        class balancing is handled by WeightedRandomSampler in the DataLoader.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            class_counts: Dict mapping class index to sample count (for logging).
            genre_names: Genre names (for per-class F1 logging).
        Returns:
            MetricsTracker with complete per-epoch history.
        """
        epochs = self.config.get("epochs", 80)
        lr = self.config.get("lr", 0.002)
        weight_decay = self.config.get("weight_decay", 1e-4)
        label_smoothing = self.config.get("label_smoothing", 0.1)
        early_stopping_patience = self.config.get("early_stopping_patience", 25)
        lr_reduce_patience = self.config.get("lr_reduce_patience", 8)
        lr_reduce_factor = self.config.get("lr_reduce_factor", 0.5)
        checkpoint_freq = self.config.get("checkpoint_freq", 10)

        self._optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self._criterion = nn.CrossEntropyLoss(
            weight=None,
            label_smoothing=label_smoothing,
        )

        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=lr_reduce_factor,
            patience=lr_reduce_patience,
        )

        self._early_stop_counter = 0

        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)
        logger.info(
            "Epochs: %d | LR: %.4f | Weight decay: %.1e | "
            "Label smoothing: %.2f | Early stopping: %d | LR patience: %d",
            epochs,
            lr,
            weight_decay,
            label_smoothing,
            early_stopping_patience,
            lr_reduce_patience,
        )
        logger.info("Class balancing: WeightedRandomSampler in DataLoader")
        logger.info("Loss: CrossEntropyLoss (no class weights)")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)
            epoch_time = time.time() - epoch_start
            current_lr = self._optimizer.param_groups[0]["lr"]

            self.tracker.update(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr,
                epoch_time=epoch_time,
            )

            improved = val_acc > self.best_val_acc
            status = "⭐" if improved else "  "

            logger.info(
                "Epoch %3d/%3d — train: loss=%.4f acc=%.4f | "
                "val: loss=%.4f acc=%.4f %s | time=%.1fs | lr=%.2e",
                epoch,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                status,
                epoch_time,
                current_lr,
            )

            if improved:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._early_stop_counter = 0
                self._save_checkpoint("best_model.pt")
            else:
                self._early_stop_counter += 1

            self._scheduler.step(val_loss)

            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self._save_intermediate(epoch)
                if class_counts is not None and genre_names is not None:
                    self._log_per_class_f1(val_loader, genre_names, class_counts)

            if self._early_stop_counter >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    self._early_stop_counter,
                )
                break

        self.tracker.save_plots(self.plots_dir / "training_curves.png")
        self.tracker.save_csv(self.metrics_dir / "training_history.csv")
        self.tracker.save_json(self.metrics_dir / "training_history.json")

        self._plot_training_summary()

        total_min = sum(self.tracker.epoch_times) / 60.0
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(
            "Best: val_acc=%.4f (epoch %d) | Total time: %.1f min",
            self.best_val_acc,
            self.best_epoch,
            total_min,
        )
        logger.info("Experiment: %s", self.exp_dir)

        return self.tracker

    def _plot_training_summary(self) -> None:
        """Generate four-panel training summary plot."""
        if not HAS_PLOT:
            logger.warning("Matplotlib not available, skipping training summary plot")
            return

        if not self.tracker.epochs:
            return

        epochs = self.tracker.epochs
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(epochs, self.tracker.train_loss, 'b-', label='Train', alpha=0.7)
        ax.plot(epochs, self.tracker.val_loss, 'r-', label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(epochs, self.tracker.train_acc, 'b-', label='Train', alpha=0.7)
        ax.plot(epochs, self.tracker.val_acc, 'r-', label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(epochs, self.tracker.learning_rates, 'g-', marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.bar(epochs, self.tracker.epoch_times, alpha=0.7, color='steelblue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Epoch Duration')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(
            f'Training Summary - Best val_acc: {self.best_val_acc:.4f} '
            f'(epoch {self.best_epoch})',
            fontsize=14, fontweight='bold',
        )
        plt.tight_layout()
        plt.savefig(self.plots_dir / "training_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Training summary plot saved: %s", self.plots_dir / "training_summary.png")

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        genre_names: List[str],
        run_deep_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate the best model on the test set.

        Args:
            test_loader: Test DataLoader.
            genre_names: Ordered list of genre name strings.
            run_deep_analysis: Run comprehensive PyTorchModelAnalyzer evaluation.

        Returns:
            Dictionary with comprehensive analysis results.
        """
        best_path = self.checkpoints_dir / "best_model.pt"
        if best_path.exists():
            logger.info("Loading best checkpoint for evaluation: %s", best_path)
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])
            logger.info(
                "Loaded model with val_acc=%.4f (epoch %d)",
                checkpoint.get("best_val_acc", 0.0),
                checkpoint.get("best_epoch", 0),
            )
        else:
            logger.warning("No best_model.pt found — evaluating current model state")

        self.model.to(self.device)
        self.model.eval()

        if run_deep_analysis:
            logger.info("=" * 60)
            logger.info("RUNNING DEEP MODEL ANALYSIS")
            logger.info("=" * 60)

            analyzer = PyTorchModelAnalyzer(
                model=self.model,
                genre_names=genre_names,
                device=self.device,
                model_name=f"AudioCNN_{self.config.get('subset', 'unknown')}",
            )

            results = analyzer.full_analysis(
                dataloader=test_loader,
                save_dir=self.analysis_dir,
                generate_plots=True,
                export_json=True,
            )

            self.analyzer = analyzer

            test_metrics = {
                "accuracy": results["metrics"].get("accuracy", 0.0),
                "balanced_accuracy": results["metrics"].get("balanced_accuracy", 0.0),
                "f1_macro": results["metrics"].get("f1_macro", 0.0),
                "f1_weighted": results["metrics"].get("f1_weighted", 0.0),
                "top_3_accuracy": results["metrics"].get("top_k", {}).get(3, 0.0),
                "top_5_accuracy": results["metrics"].get("top_k", {}).get(5, 0.0),
                "composite_score": results["metrics"].get("composite_score", 0.0),
                "mcc": results["metrics"].get("matthews_corrcoef", 0.0),
                "ece": results["metrics"].get("calibration", {}).get("ece", 0.0),
                "error_rate": results["metrics"].get("error_analysis", {}).get("error_rate", 0.0),
            }
        else:
            logger.info("Running basic evaluation...")
            results = self._basic_evaluation(test_loader, genre_names)
            test_metrics = results["metrics"]

        def convert_to_serializable(obj: Any) -> Any:
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if obj is None:
                return None
            return obj

        test_metrics_serializable = {
            k: convert_to_serializable(v) for k, v in test_metrics.items()
        }

        with open(self.metrics_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics_serializable, f, indent=2, ensure_ascii=False)

        self._save_summary(test_metrics)

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(
            "Test Results — accuracy=%.4f, f1_macro=%.4f, "
            "top-3=%.4f, composite=%.4f",
            test_metrics.get("accuracy", 0.0),
            test_metrics.get("f1_macro", 0.0),
            test_metrics.get("top_3_accuracy", 0.0),
            test_metrics.get("composite_score", 0.0),
        )

        if run_deep_analysis and "ece" in test_metrics:
            logger.info("Calibration — ECE=%.4f, Reliability=%.4f",
                        test_metrics["ece"], 1.0 - test_metrics["ece"])

        logger.info("Analysis artifacts saved to: %s", self.analysis_dir)

        return results

    def _basic_evaluation(
        self,
        test_loader: DataLoader,
        genre_names: List[str],
    ) -> Dict[str, Any]:
        """Run basic evaluation without deep analysis (fallback)."""
        all_preds = []
        all_labels = []

        logger.info("Running inference on test set...")
        for x, y in test_loader:
            x = self._prepare_input(x).to(self.device)
            out = self.model(x)
            all_preds.append(out.cpu().numpy())
            all_labels.append(y.numpy())

        y_pred_proba = np.concatenate(all_preds, axis=0)
        y_test = np.concatenate(all_labels, axis=0)
        y_pred = y_pred_proba.argmax(axis=1)

        accuracy = float(accuracy_score(y_test, y_pred))
        f1_macro = float(f1_score(y_test, y_pred, average='macro'))
        f1_weighted = float(f1_score(y_test, y_pred, average='weighted'))

        report = classification_report(
            y_test, y_pred, target_names=genre_names, output_dict=True, zero_division=0
        )

        return {
            "metrics": {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "error_rate": float(1.0 - accuracy),
            },
            "per_class": {
                genre: {
                    "precision": report[genre]["precision"],
                    "recall": report[genre]["recall"],
                    "f1": report[genre]["f1-score"],
                    "support": report[genre]["support"],
                }
                for genre in genre_names if genre in report
            },
            "y_pred": y_pred,
            "y_test": y_test,
            "y_pred_proba": y_pred_proba,
        }

    def _save_summary(self, test_metrics: Dict[str, Any]) -> None:
        """Save a structured experiment summary JSON."""
        best = self.tracker.get_best_metrics() if self.tracker.epochs else {}

        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "model": {
                "architecture": type(self.model).__name__,
                "parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                "n_mfcc": self.config.get("n_mfcc", 40),
                "n_channels": self.config.get("n_channels", 3),
                "n_classes": self.config.get("n_classes", 16),
            },
            "training_results": {
                "best_epoch": self.best_epoch,
                "best_val_acc": self.best_val_acc,
                "total_epochs": len(self.tracker.epochs),
                "total_time_minutes": (
                    sum(self.tracker.epoch_times) / 60.0
                    if self.tracker.epoch_times
                    else 0.0
                ),
            },
            "test_results": test_metrics,
            "paths": {
                "checkpoint": str(self.checkpoints_dir / "best_model.pt"),
                "training_plots": str(self.plots_dir / "training_curves.png"),
                "analysis_dir": str(self.analysis_dir),
                "metrics_csv": str(self.metrics_dir / "training_history.csv"),
                "test_metrics": str(self.metrics_dir / "test_metrics.json"),
            },
        }

        with open(self.exp_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Experiment summary saved: %s", self.exp_dir / "summary.json")

    def load_best_model(self) -> None:
        """Load the best saved checkpoint into the model."""
        best_path = self.checkpoints_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best model not found: {best_path}")

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)

        logger.info(
            "Best model loaded: %s (val_acc=%.4f, epoch=%d)",
            best_path,
            self.best_val_acc,
            self.best_epoch,
        )

    def print_info(self) -> None:
        """Print a human-readable training summary."""
        if not self.tracker.epochs:
            print("No training data yet. Run fit() first.")
            return

        best = self.tracker.get_best_metrics()
        final = self.tracker.get_final_metrics()
        total_min = sum(self.tracker.epoch_times) / 60.0

        print("=" * 70)
        print("TRAINING SUMMARY — AudioCNN")
        print("=" * 70)
        print(f"Experiment:     {self.experiment_name}")
        print(f"Directory:      {self.exp_dir}")
        print(f"Epochs:         {len(self.tracker.epochs)} completed")
        print(f"Total time:     {total_min:.1f} min "
              f"({np.mean(self.tracker.epoch_times):.1f} sec/epoch)")
        print(f"Best epoch:     {best['epoch']} "
              f"(val_loss={best['val_loss']:.4f}, val_acc={best['val_acc']:.4f})")
        print(f"Final epoch:    {final['epoch']} "
              f"(val_loss={final['val_loss']:.4f}, val_acc={final['val_acc']:.4f})")
        print(f"Learning rate:  {final['lr']:.2e}")
        print("-" * 70)
        print("Artifacts:")
        print(f"  Checkpoint:   {self.checkpoints_dir / 'best_model.pt'}")
        print(f"  Plots:        {self.plots_dir}")
        print(f"  Metrics:      {self.metrics_dir}")
        print(f"  Analysis:     {self.analysis_dir}")
        print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for CNN training."""
    parser = argparse.ArgumentParser(
        description="Train AudioCNN for MFCC genre classification"
    )
    parser.add_argument("--subset", type=str, default="medium",
                        choices=["small", "medium", "large"], help="FMA subset")
    parser.add_argument("--min_samples", type=int, default=10,
                        help="Minimum tracks per genre")
    parser.add_argument("--dataset_id", type=str, default=None,
                        help="Specific dataset ID from pipeline cache")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Initial learning rate")
    parser.add_argument("--target_frames", type=int, default=430,
                        help="MFCC time frames per sample (430 ≈ 10s)")
    parser.add_argument("--no_deltas", action="store_true",
                        help="Disable delta features")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable augmentation for training")
    parser.add_argument("--no_balanced_sampling", action="store_true",
                        help="Disable balanced sampling (use shuffle instead)")
    parser.add_argument("--n_mfcc", type=int, default=40,
                        help="Number of MFCC coefficients")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout after conv stem")
    parser.add_argument("--fc_dropout", type=float, default=0.2,
                        help="Dropout between FC layers")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 weight decay")
    parser.add_argument("--early_stopping", type=int, default=25,
                        help="Early stopping patience")
    parser.add_argument("--lr_patience", type=int, default=8,
                        help="LR reduction patience")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="LR reduction factor")
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                        help="Save intermediate checkpoint every N epochs")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="Gradient clipping norm")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Custom experiment directory name")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--skip_deep_analysis", action="store_true",
                        help="Skip deep model analysis")
    return parser


def main() -> Dict[str, Any]:
    """Main entry point for CNN training."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level, mode="both", console_level=args.log_level)

    use_deltas = not args.no_deltas
    n_channels = 3 if use_deltas else 1

    logger.info("=" * 60)
    logger.info("AUDIO CNN TRAINING")
    logger.info("=" * 60)
    logger.info(
        "Config — subset=%s, min_samples=%d, epochs=%d, batch_size=%d, "
        "target_frames=%d, deltas=%s, channels=%d, augment=%s, "
        "balanced_sampling=%s, lr=%.4f, dropout=%.2f/%.2f, "
        "label_smoothing=%.2f, n_mfcc=%d",
        args.subset,
        args.min_samples,
        args.epochs,
        args.batch_size,
        args.target_frames,
        use_deltas,
        n_channels,
        not args.no_augment,
        not args.no_balanced_sampling,
        args.lr,
        args.dropout,
        args.fc_dropout,
        args.label_smoothing,
        args.n_mfcc,
    )

    logger.info("Loading data...")
    train_loader, val_loader, test_loader, genre_names, failure_report, class_counts = \
        create_mfcc_dataloaders(
            subset=args.subset,
            min_samples_per_genre=args.min_samples,
            dataset_id=args.dataset_id,
            batch_size=args.batch_size,
            target_frames=args.target_frames,
            use_deltas=use_deltas,
            augment_train=not args.no_augment,
            use_balanced_sampling=not args.no_balanced_sampling,
            n_mfcc=args.n_mfcc,
            num_workers=0,
        )

    n_classes = len(genre_names)
    logger.info("Genres: %d | Train batches: %d | Val: %d | Test: %d",
                n_classes, len(train_loader), len(val_loader), len(test_loader))

    total_failed = (
        failure_report["train"]["failed_count"]
        + failure_report["val"]["failed_count"]
        + failure_report["test"]["failed_count"]
    )
    if total_failed > 0:
        logger.warning(
            "Failed tracks — train: %d, val: %d, test: %d",
            failure_report["train"]["failed_count"],
            failure_report["val"]["failed_count"],
            failure_report["test"]["failed_count"],
        )

    # Log class distribution with rarity categories
    rare_threshold = 100
    medium_threshold = 500
    rare_count = sum(1 for c in class_counts.values() if c <= rare_threshold)
    medium_count = sum(1 for c in class_counts.values() if rare_threshold < c <= medium_threshold)
    common_count = sum(1 for c in class_counts.values() if c > medium_threshold)
    logger.info(
        "Class distribution — rare (≤%d): %d, medium (≤%d): %d, common: %d",
        rare_threshold, rare_count, medium_threshold, medium_count, common_count,
    )

    logger.info("Creating model...")
    model = build_model(
        n_mfcc=args.n_mfcc,
        n_channels=n_channels,
        n_classes=n_classes,
        dropout=args.dropout,
        fc_dropout=args.fc_dropout,
        target_frames=args.target_frames,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model on device: %s | Parameters: %d", device, n_params)

    config = {
        "subset": args.subset,
        "min_samples": args.min_samples,
        "n_channels": n_channels,
        "n_mfcc": args.n_mfcc,
        "target_frames": args.target_frames,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "fc_dropout": args.fc_dropout,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping,
        "lr_reduce_patience": args.lr_patience,
        "lr_reduce_factor": args.lr_factor,
        "checkpoint_freq": args.checkpoint_freq,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "use_deltas": use_deltas,
        "augment": not args.no_augment,
        "use_balanced_sampling": not args.no_balanced_sampling,
        "dataset_id": args.dataset_id,
        "n_classes": n_classes,
        "genre_names": genre_names,
    }

    trainer = CNNTrainer(
        model=model,
        device=device,
        config=config,
        experiment_name=args.experiment_name,
    )

    trainer.fit(train_loader, val_loader, class_counts, genre_names)

    results = trainer.evaluate(
        test_loader=test_loader,
        genre_names=genre_names,
        run_deep_analysis=not args.skip_deep_analysis,
    )

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info("Experiment:  %s", trainer.experiment_name)
    logger.info("Best epoch:  %d (val_acc=%.4f)", trainer.best_epoch, trainer.best_val_acc)
    logger.info("Test acc:    %.4f", results["metrics"].get("accuracy", 0.0))
    logger.info("Test f1:     %.4f", results["metrics"].get("f1_macro", 0.0))

    if "top_3_accuracy" in results["metrics"]:
        logger.info("Top-3 acc:   %.4f", results["metrics"]["top_3_accuracy"])
    if "ece" in results["metrics"]:
        logger.info("ECE:         %.4f", results["metrics"]["ece"])

    logger.info("Artifacts:   %s", trainer.exp_dir)

    return results


def main_parametrized(
    subset: str = "medium",
    min_samples: int = 10,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 0.002,
    target_frames: int = 430,
    use_deltas: bool = True,
    use_augment: bool = True,
    use_balanced_sampling: bool = True,
    n_mfcc: int = 40,
    dropout: float = 0.3,
    fc_dropout: float = 0.2,
    weight_decay: float = 1e-4,
    early_stopping: int = 25,
    lr_patience: int = 8,
    lr_factor: float = 0.5,
    checkpoint_freq: int = 10,
    label_smoothing: float = 0.1,
    grad_clip: Optional[float] = None,
    experiment_name: Optional[str] = None,
    log_level: str = "INFO",
    skip_deep_analysis: bool = False,
    dataset_id: Optional[str] = None,
    use_spectral_features: bool = False,
    use_chroma: bool = False,
) -> Dict[str, Any]:
    """
    Parameterized main function for CNN training.

    Args:
        subset: FMA subset ('small', 'medium', 'large').
        min_samples: Minimum tracks per genre.
        epochs: Maximum training epochs (default: 80).
        batch_size: Batch size (default: 32).
        lr: Initial learning rate (default: 0.002).
        target_frames: MFCC time frames per sample (default: 430 ≈ 10s).
        use_deltas: Include delta and delta-delta features.
        use_augment: Enable augmentation for training.
        use_balanced_sampling: Use WeightedRandomSampler.
        n_mfcc: Number of MFCC coefficients (default: 40).
        dropout: Dropout after conv stem (default: 0.3).
        fc_dropout: Dropout between FC layers (default: 0.2).
        weight_decay: L2 weight decay.
        early_stopping: Early stopping patience (default: 25).
        lr_patience: LR reduction patience (default: 8).
        lr_factor: LR reduction factor.
        checkpoint_freq: Save checkpoint every N epochs.
        label_smoothing: Label smoothing factor (default: 0.1).
        grad_clip: Gradient clipping norm.
        experiment_name: Custom experiment directory name.
        log_level: Logging level.
        skip_deep_analysis: Skip deep model analysis.
        dataset_id: Specific dataset ID from pipeline cache.
        use_spectral_features: Add spectral features.
        use_chroma: Add chroma features.

    Returns:
        Dictionary with test evaluation results.
    """
    setup_logging(level=log_level, mode="both", console_level=log_level)

    n_channels = 3 if use_deltas else 1

    logger.info("=" * 60)
    logger.info("AUDIO CNN TRAINING (Programmatic)")
    logger.info("=" * 60)
    logger.info(
        "Config — subset=%s, min_samples=%d, epochs=%d, batch_size=%d, "
        "target_frames=%d, deltas=%s, channels=%d, augment=%s, "
        "balanced_sampling=%s, n_mfcc=%d, spectral=%s, chroma=%s, "
        "lr=%.4f, dropout=%.2f/%.2f, label_smoothing=%.2f",
        subset, min_samples, epochs, batch_size,
        target_frames, use_deltas, n_channels, use_augment,
        use_balanced_sampling, n_mfcc, use_spectral_features, use_chroma,
        lr, dropout, fc_dropout, label_smoothing,
    )

    logger.info("Loading data...")
    train_loader, val_loader, test_loader, genre_names, failure_report, class_counts = \
        create_mfcc_dataloaders(
            subset=subset,
            min_samples_per_genre=min_samples,
            dataset_id=dataset_id,
            batch_size=batch_size,
            target_frames=target_frames,
            use_deltas=use_deltas,
            augment_train=use_augment,
            use_balanced_sampling=use_balanced_sampling,
            n_mfcc=n_mfcc,
            use_spectral_features=use_spectral_features,
            use_chroma=use_chroma,
            num_workers=0,
        )

    n_classes = len(genre_names)
    logger.info("Genres: %d | Train batches: %d | Val: %d | Test: %d",
                n_classes, len(train_loader), len(val_loader), len(test_loader))

    total_failed = (
        failure_report["train"]["failed_count"]
        + failure_report["val"]["failed_count"]
        + failure_report["test"]["failed_count"]
    )
    if total_failed > 0:
        logger.warning(
            "Failed tracks — train: %d, val: %d, test: %d",
            failure_report["train"]["failed_count"],
            failure_report["val"]["failed_count"],
            failure_report["test"]["failed_count"],
        )

    # Log class distribution with rarity categories
    rare_threshold = 100
    medium_threshold = 500
    rare_count = sum(1 for c in class_counts.values() if c <= rare_threshold)
    medium_count = sum(1 for c in class_counts.values() if rare_threshold < c <= medium_threshold)
    common_count = sum(1 for c in class_counts.values() if c > medium_threshold)
    logger.info(
        "Class distribution — rare (≤%d): %d, medium (≤%d): %d, common: %d",
        rare_threshold, rare_count, medium_threshold, medium_count, common_count,
    )

    logger.info("Creating model...")
    model = build_model(
        n_mfcc=n_mfcc,
        n_channels=n_channels,
        n_classes=n_classes,
        dropout=dropout,
        fc_dropout=fc_dropout,
        target_frames=target_frames,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model on device: %s | Parameters: %d", device, n_params)

    config = {
        "subset": subset,
        "min_samples": min_samples,
        "n_channels": n_channels,
        "n_mfcc": n_mfcc,
        "target_frames": target_frames,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "dropout": dropout,
        "fc_dropout": fc_dropout,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping,
        "lr_reduce_patience": lr_patience,
        "lr_reduce_factor": lr_factor,
        "checkpoint_freq": checkpoint_freq,
        "label_smoothing": label_smoothing,
        "grad_clip": grad_clip,
        "use_deltas": use_deltas,
        "augment": use_augment,
        "use_balanced_sampling": use_balanced_sampling,
        "dataset_id": dataset_id,
        "n_classes": n_classes,
        "genre_names": genre_names,
        "use_spectral_features": use_spectral_features,
        "use_chroma": use_chroma,
    }

    trainer = CNNTrainer(
        model=model,
        device=device,
        config=config,
        experiment_name=experiment_name,
    )

    trainer.fit(train_loader, val_loader, class_counts, genre_names)

    results = trainer.evaluate(
        test_loader=test_loader,
        genre_names=genre_names,
        run_deep_analysis=not skip_deep_analysis,
    )

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info("Experiment:  %s", trainer.experiment_name)
    logger.info("Best epoch:  %d (val_acc=%.4f)", trainer.best_epoch, trainer.best_val_acc)
    logger.info("Test acc:    %.4f", results["metrics"].get("accuracy", 0.0))
    logger.info("Test f1:     %.4f", results["metrics"].get("f1_macro", 0.0))

    if "top_3_accuracy" in results["metrics"]:
        logger.info("Top-3 acc:   %.4f", results["metrics"]["top_3_accuracy"])
    if "ece" in results["metrics"]:
        logger.info("ECE:         %.4f", results["metrics"]["ece"])

    logger.info("Artifacts:   %s", trainer.exp_dir)

    return results


if __name__ == "__main__":
    main_parametrized(
        subset="medium",
        min_samples=10,
        epochs=80,
        lr=0.002,
        target_frames=430,
        n_mfcc=40,
        dropout=0.3,
        fc_dropout=0.2,
        label_smoothing=0.1,
        use_balanced_sampling=True,
    )