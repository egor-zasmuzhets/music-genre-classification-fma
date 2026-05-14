"""
src/training/train_cnn_debug.py
Minimal CNN training script for pipeline validation and debugging.

Trains a lightweight MiniCNN on MFCC features to verify that the full
data pipeline — from cached features through DataLoaders to model
checkpointing — functions correctly before running full experiments.

Typical usage:
    python -m src.training.train_cnn_debug

    # Custom configuration
    result = debug_train(
        subset="medium",
        min_samples=10,
        epochs=15,
        batch_size=32
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.torch_dataset import create_mfcc_dataloaders
from src.models.cnn_mfcc_debug import MiniCNN
from src.training.metrics_tracker_debug import DebugTracker, check_dataloader
from src.utils.config import paths
from src.utils.logging_utils import setup_logging


logger = logging.getLogger(__name__)


# ============================================================================
# TRAINING & VALIDATION LOOPS
# ============================================================================

def _prepare_input(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize input tensor shape for MiniCNN.

    The DataLoader may produce (B, n_mfcc, 1, frames) — the channel
    dimension inserted by MFCCDataset. MiniCNN expects (B, n_mfcc, frames)
    and adds its own channel dim internally.

    Args:
        x: Input tensor from DataLoader, 3D or 4D.

    Returns:
        Tensor of shape (B, n_mfcc, frames).
    """
    if x.dim() == 4 and x.shape[2] == 1:
        x = x.squeeze(2)
    return x


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Run a single training epoch.

    Args:
        model: The CNN model in training mode.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Torch device string ('cpu' or 'cuda').

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = _prepare_input(x).to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Evaluate the model on a validation or test set.

    Args:
        model: The CNN model in eval mode.
        loader: Validation/Test DataLoader.
        criterion: Loss function.
        device: Torch device string.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = _prepare_input(x).to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


# ============================================================================
# PER-CLASS ACCURACY
# ============================================================================

@torch.no_grad()
def compute_per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    n_classes: int,
    device: str,
) -> torch.Tensor:
    """
    Compute per-class accuracy on a dataset.

    Args:
        model: The CNN model in eval mode.
        loader: DataLoader for evaluation.
        n_classes: Total number of classes.
        device: Torch device string.

    Returns:
        1D tensor of per-class accuracy values (length n_classes).
    """
    model.eval()
    class_correct = torch.zeros(n_classes)
    class_total = torch.zeros(n_classes)

    for x, y in loader:
        x = _prepare_input(x).to(device)
        y = y.to(device)

        out = model(x)
        _, pred = out.max(1)

        for i in range(len(y)):
            label = y[i]
            class_total[label] += 1
            if pred[i] == label:
                class_correct[label] += 1

    per_class_acc = torch.where(
        class_total > 0,
        class_correct / class_total,
        torch.zeros_like(class_total),
    )

    return per_class_acc


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def debug_train(
    subset: str = "small",
    min_samples: int = 50,
    epochs: int = 10,
    batch_size: int = 16,
    target_frames: int = 64,
    use_deltas: bool = False,
    learning_rate: float = 0.001,
) -> Dict[str, Any]:
    """
    Run a minimal end-to-end training for pipeline debugging.

    Loads data, creates DataLoaders, trains a MiniCNN, evaluates on
    the test set, and saves checkpoints, plots, and config.

    Args:
        subset: FMA subset ('small', 'medium', 'large').
        min_samples: Minimum tracks per genre.
        epochs: Number of training epochs.
        batch_size: Samples per batch.
        target_frames: Time frames per MFCC sample.
        use_deltas: Include delta and delta-delta coefficients.
        learning_rate: Adam learning rate.

    Returns:
        Dictionary with keys: best_val_acc, test_acc, params,
        classes, save_dir, config.
    """
    logger.info("=" * 60)
    logger.info("DEBUG CNN TRAINING — PIPELINE VALIDATION")
    logger.info("=" * 60)
    logger.info(
        "Config — subset=%s, min_samples=%d, epochs=%d, batch_size=%d, "
        "target_frames=%d, deltas=%s, lr=%.4f",
        subset,
        min_samples,
        epochs,
        batch_size,
        target_frames,
        use_deltas,
        learning_rate,
    )

    logger.info("Creating DataLoaders...")
    train_loader, val_loader, test_loader, genres, report = \
        create_mfcc_dataloaders(
            subset=subset,
            min_samples_per_genre=min_samples,
            batch_size=batch_size,
            target_frames=target_frames,
            use_deltas=use_deltas,
            augment_train=True,
            num_workers=0,
        )

    n_classes = len(genres)
    n_train = len(train_loader.dataset)
    logger.info(
        "DataLoaders ready — %d classes, %d train samples",
        n_classes,
        n_train,
    )

    logger.info("Validating DataLoaders...")
    train_batch = check_dataloader(train_loader, "train")
    val_batch = check_dataloader(val_loader, "val")

    sample_x = train_batch[0]
    if sample_x.dim() == 4 and sample_x.shape[2] == 1:
        n_mfcc = sample_x.shape[1]
    elif sample_x.dim() == 3:
        n_mfcc = sample_x.shape[1]
    else:
        n_mfcc = 20

    logger.info("Detected n_mfcc=%d from input shape %s", n_mfcc, tuple(sample_x.shape))

    model = MiniCNN(n_mfcc=n_mfcc, n_classes=n_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "MiniCNN created — %d params, device=%s, input=(%d mfcc, %d frames)",
        n_params,
        device,
        n_mfcc,
        target_frames,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    tracker = DebugTracker()

    save_dir = paths.cnn.checkpoints_dir / "debug"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training — %d epochs", epochs)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        tracker.update(val_loss, val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        status = "⭐" if improved else "  "
        logger.info(
            "Epoch %2d/%d — train: loss=%.4f acc=%.4f | "
            "val: loss=%.4f acc=%.4f %s",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            status,
        )

    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    per_class_acc = compute_per_class_accuracy(
        model, test_loader, n_classes, device,
    )

    logger.info("Test results — loss=%.4f, accuracy=%.4f", test_loss, test_acc)

    for i, genre in enumerate(genres):
        n_samples = int(
            sum(1 for _, y in test_loader for label in y if label == i)
        )
        logger.info(
            "  %-20s: %.3f (%d samples)",
            genre,
            per_class_acc[i].item(),
            n_samples,
        )

    tracker.save_plots(save_dir / "debug_plot.png")
    tracker.save_data(save_dir / "debug_metrics.json")

    config = {
        "subset": subset,
        "min_samples": min_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "target_frames": target_frames,
        "use_deltas": use_deltas,
        "learning_rate": learning_rate,
        "n_mfcc": n_mfcc,
        "n_classes": n_classes,
        "model_params": n_params,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "classes": genres,
    }

    with open(save_dir / "debug_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("All artifacts saved to: %s", save_dir)
    logger.info(
        "Final results — best_val_acc=%.4f, test_acc=%.4f",
        best_val_acc,
        test_acc,
    )

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "params": n_params,
        "classes": genres,
        "save_dir": str(save_dir),
        "config": config,
    }


# ============================================================================
# MAIN GUARD
# ============================================================================

if __name__ == "__main__":
    setup_logging(level="INFO", mode="console")

    result = debug_train(
        subset="medium",
        min_samples=50,
        epochs=3,
        batch_size=16,
        target_frames=64,
        use_deltas=False,
    )

    save_dir = Path(result["save_dir"])
    metrics_file = save_dir / "debug_metrics.json"

    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Best val acc:  {max(data['accuracy']):.4f}")
        print(f"Test acc:      {result['test_acc']:.4f}")
        print(f"Classes:       {len(result['classes'])}")
        print(f"Model params:  {result['params']:,}")
        print(f"Saved to:      {save_dir}")
        print("=" * 60)