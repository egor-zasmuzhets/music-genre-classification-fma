#!/usr/bin/env python
"""
CNN training script for MFCC-based audio genre classification.

Orchestrates data loading, model creation, training with early stopping,
and comprehensive evaluation with PyTorchModelAnalyzer. All artifacts are
saved into a timestamped experiment directory for reproducibility.

Uses balanced sampling instead of class weights for imbalanced data.

V2 Changes:
- Uses updated data loading (V2 format)
- No changes to core training logic
- Works with new directory structure

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
    """Create an AudioCNN model with the specified configuration."""
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

    Uses balanced sampling via WeightedRandomSampler in DataLoader.

    V2: Works with new directory structure, no pickle dependencies.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        config: Dict[str, Any],
        experiment_name: Optional[str] = None,
    ) -> None:
        """Initialize the trainer."""
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

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is 4D (B, C, H, W) for Conv2d."""
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
            "Epoch %d — checkpoint | best: val_acc=%.4f (epoch %d) | current: val_acc=%.4f | lr=%.2e",
            epoch, best["val_acc"], best["epoch"], current_val_acc, current_lr
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
        """Log per-class F1 scores grouped by rarity."""
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

        logger.info("--- Per-class F1 (Rare ≤%d | Medium ≤%d) ---", rare_threshold, medium_threshold)

        if rare_f1s:
            rare_str = "  ".join(f"{g}:{f:.3f}" for g, f, _ in rare_f1s[:5])
            logger.info("  Rare:   %s", rare_str)
            if len(rare_f1s) > 5:
                logger.info("    ... and %d more rare genres", len(rare_f1s) - 5)

        logger.info("  F1 → Rare: %.3f | Medium: %.3f | Common: %.3f", rare_mean, medium_mean, common_mean)

        if rare_f1s:
            worst_genre, worst_f1, worst_count = min(rare_f1s, key=lambda x: x[1])
            logger.info("  ⚠ Worst rare: %s (F1=%.3f, %d samples)", worst_genre, worst_f1, worst_count)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_counts: Optional[Dict[int, int]] = None,
        genre_names: Optional[List[str]] = None,
    ) -> MetricsTracker:
        """Run the full training loop."""
        epochs = self.config.get("epochs", 80)
        lr = self.config.get("lr", 0.002)
        weight_decay = self.config.get("weight_decay", 1e-4)
        label_smoothing = self.config.get("label_smoothing", 0.1)
        early_stopping_patience = self.config.get("early_stopping_patience", 25)
        lr_reduce_patience = self.config.get("lr_reduce_patience", 8)
        lr_reduce_factor = self.config.get("lr_reduce_factor", 0.5)
        checkpoint_freq = self.config.get("checkpoint_freq", 10)

        self._optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self._criterion = nn.CrossEntropyLoss(weight=None, label_smoothing=label_smoothing)
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode="min", factor=lr_reduce_factor, patience=lr_reduce_patience
        )
        self._early_stop_counter = 0

        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)
        logger.info(
            "Epochs: %d | LR: %.4f | Weight decay: %.1e | Label smoothing: %.2f | "
            "Early stopping: %d | LR patience: %d",
            epochs, lr, weight_decay, label_smoothing, early_stopping_patience, lr_reduce_patience
        )
        logger.info("Class balancing: WeightedRandomSampler in DataLoader")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)
            epoch_time = time.time() - epoch_start
            current_lr = self._optimizer.param_groups[0]["lr"]

            self.tracker.update(
                epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss, val_acc=val_acc, lr=current_lr, epoch_time=epoch_time
            )

            improved = val_acc > self.best_val_acc
            status = "⭐" if improved else "  "

            logger.info(
                "Epoch %3d/%3d — train: loss=%.4f acc=%.4f | val: loss=%.4f acc=%.4f %s | time=%.1fs | lr=%.2e",
                epoch, epochs, train_loss, train_acc, val_loss, val_acc, status, epoch_time, current_lr
            )

            if improved:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._early_stop_counter = 0
                self._save_checkpoint("cnn_model.pt")
            else:
                self._early_stop_counter += 1

            self._scheduler.step(val_loss)

            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self._save_intermediate(epoch)
                if class_counts is not None and genre_names is not None:
                    self._log_per_class_f1(val_loader, genre_names, class_counts)

            if self._early_stop_counter >= early_stopping_patience:
                logger.info("Early stopping after %d epochs without improvement", self._early_stop_counter)
                break

        self.tracker.save_plots(self.plots_dir / "training_curves.png")
        self.tracker.save_csv(self.metrics_dir / "training_history.csv")
        self.tracker.save_json(self.metrics_dir / "training_history.json")
        self._plot_training_summary()

        total_min = sum(self.tracker.epoch_times) / 60.0
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("Best: val_acc=%.4f (epoch %d) | Total time: %.1f min", self.best_val_acc, self.best_epoch, total_min)

        return self.tracker

    def _plot_training_summary(self) -> None:
        """Generate four-panel training summary plot."""
        if not HAS_PLOT or not self.tracker.epochs:
            return

        epochs = self.tracker.epochs
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(epochs, self.tracker.train_loss, 'b-', label='Train', alpha=0.7)
        axes[0, 0].plot(epochs, self.tracker.val_loss, 'r-', label='Val', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, self.tracker.train_acc, 'b-', label='Train', alpha=0.7)
        axes[0, 1].plot(epochs, self.tracker.val_acc, 'r-', label='Val', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Convergence')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, self.tracker.learning_rates, 'g-', marker='o', markersize=4)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].bar(epochs, self.tracker.epoch_times, alpha=0.7, color='steelblue')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Epoch Duration')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Training Summary - Best val_acc: {self.best_val_acc:.4f} (epoch {self.best_epoch})')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "training_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Training summary plot saved")

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        genre_names: List[str],
        run_deep_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate the best model on the test set."""
        best_path = self.checkpoints_dir / "cnn_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])
            logger.info("Loaded best model: val_acc=%.4f (epoch %d)", checkpoint.get("best_val_acc", 0.0), checkpoint.get("best_epoch", 0))
        else:
            logger.warning("No cnn_model.pt found — evaluating current model")

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

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        with open(self.metrics_dir / "test_metrics.json", "w") as f:
            json.dump({k: convert(v) for k, v in test_metrics.items()}, f, indent=2)

        self._save_summary(test_metrics)

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(
            "Test Results — accuracy=%.4f, f1_macro=%.4f, top-3=%.4f, composite=%.4f",
            test_metrics.get("accuracy", 0.0), test_metrics.get("f1_macro", 0.0),
            test_metrics.get("top_3_accuracy", 0.0), test_metrics.get("composite_score", 0.0)
        )

        return results

    def _basic_evaluation(
        self,
        test_loader: DataLoader,
        genre_names: List[str],
    ) -> Dict[str, Any]:
        """Run basic evaluation without deep analysis."""
        all_preds, all_labels = [], []

        for x, y in test_loader:
            x = self._prepare_input(x).to(self.device)
            out = self.model(x)
            all_preds.append(out.cpu().numpy())
            all_labels.append(y.numpy())

        y_pred_proba = np.concatenate(all_preds, axis=0)
        y_test = np.concatenate(all_labels, axis=0)
        y_pred = y_pred_proba.argmax(axis=1)

        return {
            "metrics": {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average='macro')),
                "f1_weighted": float(f1_score(y_test, y_pred, average='weighted')),
                "error_rate": float(1.0 - accuracy_score(y_test, y_pred)),
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
                "parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "n_mfcc": self.config.get("n_mfcc", 40),
                "n_channels": self.config.get("n_channels", 3),
                "n_classes": self.config.get("n_classes", 16),
            },
            "training_results": {
                "best_epoch": self.best_epoch,
                "best_val_acc": self.best_val_acc,
                "total_epochs": len(self.tracker.epochs),
                "total_time_minutes": sum(self.tracker.epoch_times) / 60.0 if self.tracker.epoch_times else 0.0,
            },
            "test_results": test_metrics,
            "paths": {
                "checkpoint": str(self.checkpoints_dir / "cnn_model.pt"),
                "analysis_dir": str(self.analysis_dir),
            },
            "format_version": 2,
        }

        with open(self.exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Experiment summary saved: %s", self.exp_dir / "summary.json")

    def load_best_model(self) -> None:
        """Load the best saved checkpoint into the model."""
        best_path = self.checkpoints_dir / "cnn_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best model not found: {best_path}")

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        logger.info("Best model loaded: val_acc=%.4f (epoch %d)", self.best_val_acc, self.best_epoch)

    def print_info(self) -> None:
        """Print a human-readable training summary."""
        if not self.tracker.epochs:
            print("No training data yet. Run fit() first.")
            return

        best = self.tracker.get_best_metrics()
        final = self.tracker.get_final_metrics()
        total_min = sum(self.tracker.epoch_times) / 60.0

        print("=" * 70)
        print("TRAINING SUMMARY — AudioCNN (V2)")
        print("=" * 70)
        print(f"Experiment:     {self.experiment_name}")
        print(f"Epochs:         {len(self.tracker.epochs)} completed")
        print(f"Total time:     {total_min:.1f} min ({np.mean(self.tracker.epoch_times):.1f} sec/epoch)")
        print(f"Best epoch:     {best['epoch']} (val_acc={best['val_acc']:.4f})")
        print(f"Final epoch:    {final['epoch']} (val_acc={final['val_acc']:.4f})")
        print("-" * 70)
        print(f"Checkpoint:     {self.checkpoints_dir / 'cnn_model.pt'}")
        print(f"Analysis:       {self.analysis_dir}")
        print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for CNN training."""
    parser = argparse.ArgumentParser(description="Train AudioCNN for MFCC genre classification")
    parser.add_argument("--subset", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--target_frames", type=int, default=430)
    parser.add_argument("--no_deltas", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--n_mfcc", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stopping", type=int, default=25)
    parser.add_argument("--lr_patience", type=int, default=8)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--skip_deep_analysis", action="store_true")
    return parser


def main() -> Dict[str, Any]:
    """Main entry point for CNN training."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level, mode="both", console_level=args.log_level)

    use_deltas = not args.no_deltas
    n_channels = 3 if use_deltas else 1

    logger.info("=" * 60)
    logger.info("AUDIO CNN TRAINING (V2)")
    logger.info("=" * 60)
    logger.info(
        "Config — subset=%s, min_samples=%d, epochs=%d, batch_size=%d, "
        "target_frames=%d, deltas=%s, channels=%d, augment=%s, balanced_sampling=%s, n_mfcc=%d",
        args.subset, args.min_samples, args.epochs, args.batch_size,
        args.target_frames, use_deltas, n_channels, not args.no_augment,
        not args.no_balanced_sampling, args.n_mfcc
    )

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
    logger.info("Genres: %d | Train: %d, Val: %d, Test: %d", n_classes, len(train_loader), len(val_loader), len(test_loader))

    model = build_model(
        n_mfcc=args.n_mfcc, n_channels=n_channels, n_classes=n_classes,
        dropout=args.dropout, fc_dropout=args.fc_dropout, target_frames=args.target_frames
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info("Model on device: %s | Parameters: %d", device, sum(p.numel() for p in model.parameters()))

    config = {
        "subset": args.subset, "min_samples": args.min_samples, "n_channels": n_channels,
        "n_mfcc": args.n_mfcc, "target_frames": args.target_frames, "batch_size": args.batch_size,
        "lr": args.lr, "epochs": args.epochs, "dropout": args.dropout, "fc_dropout": args.fc_dropout,
        "weight_decay": args.weight_decay, "early_stopping_patience": args.early_stopping,
        "lr_reduce_patience": args.lr_patience, "lr_reduce_factor": args.lr_factor,
        "checkpoint_freq": args.checkpoint_freq, "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip, "use_deltas": use_deltas, "augment": not args.no_augment,
        "use_balanced_sampling": not args.no_balanced_sampling, "dataset_id": args.dataset_id,
        "n_classes": n_classes, "genre_names": genre_names,
    }

    trainer = CNNTrainer(model=model, device=device, config=config, experiment_name=args.experiment_name)
    trainer.fit(train_loader, val_loader, class_counts, genre_names)

    results = trainer.evaluate(test_loader, genre_names, run_deep_analysis=not args.skip_deep_analysis)

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info("Experiment:  %s", trainer.experiment_name)
    logger.info("Best epoch:  %d (val_acc=%.4f)", trainer.best_epoch, trainer.best_val_acc)
    logger.info("Test acc:    %.4f", results["metrics"].get("accuracy", 0.0))
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
    """Parameterized main function for programmatic CNN training."""
    setup_logging(level=log_level, mode="both", console_level=log_level)

    n_channels = 3 if use_deltas else 1

    logger.info("=" * 60)
    logger.info("AUDIO CNN TRAINING (Programmatic, V2)")
    logger.info("=" * 60)

    train_loader, val_loader, test_loader, genre_names, failure_report, class_counts = \
        create_mfcc_dataloaders(
            subset=subset, min_samples_per_genre=min_samples, dataset_id=dataset_id,
            batch_size=batch_size, target_frames=target_frames, use_deltas=use_deltas,
            augment_train=use_augment, use_balanced_sampling=use_balanced_sampling,
            n_mfcc=n_mfcc, use_spectral_features=use_spectral_features, use_chroma=use_chroma,
            num_workers=0,
        )

    n_classes = len(genre_names)

    model = build_model(
        n_mfcc=n_mfcc, n_channels=n_channels, n_classes=n_classes,
        dropout=dropout, fc_dropout=fc_dropout, target_frames=target_frames
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    config = {
        "subset": subset, "min_samples": min_samples, "n_channels": n_channels,
        "n_mfcc": n_mfcc, "target_frames": target_frames, "batch_size": batch_size,
        "lr": lr, "epochs": epochs, "dropout": dropout, "fc_dropout": fc_dropout,
        "weight_decay": weight_decay, "early_stopping_patience": early_stopping,
        "lr_reduce_patience": lr_patience, "lr_reduce_factor": lr_factor,
        "checkpoint_freq": checkpoint_freq, "label_smoothing": label_smoothing,
        "grad_clip": grad_clip, "use_deltas": use_deltas, "augment": use_augment,
        "use_balanced_sampling": use_balanced_sampling, "dataset_id": dataset_id,
        "n_classes": n_classes, "genre_names": genre_names,
        "use_spectral_features": use_spectral_features, "use_chroma": use_chroma,
    }

    trainer = CNNTrainer(model=model, device=device, config=config, experiment_name=experiment_name)
    trainer.fit(train_loader, val_loader, class_counts, genre_names)

    results = trainer.evaluate(test_loader, genre_names, run_deep_analysis=not skip_deep_analysis)

    return results


if __name__ == "__main__":
    main_parametrized(
        subset="medium", min_samples=10, epochs=80, lr=0.002,
        target_frames=430, n_mfcc=40, dropout=0.3, fc_dropout=0.2,
        label_smoothing=0.1, use_balanced_sampling=True,
    )