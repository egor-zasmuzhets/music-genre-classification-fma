"""
Minimal CNN model for data pipeline debugging and rapid prototyping.

A lightweight 2D convolutional architecture that works with MFCC inputs
of shape (B, n_mfcc, frames). Designed for fast iteration during
pipeline development — not intended for production training.

Typical usage:
    from src.models.cnn_mfcc_debug import MiniCNN

    model = MiniCNN(n_mfcc=20, n_classes=16)
    output = model(mfcc_batch)  # (B, 20, 64) -> (B, 16)

    model.save(Path("checkpoints/debug_model.pt"))
    model = MiniCNN.load(Path("checkpoints/debug_model.pt"))
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class MiniCNN(nn.Module):
    """
    Minimal 2D CNN for MFCC-based genre classification debugging.

    Architecture:
        Input:  (B, n_mfcc, frames)
        Reshape: (B, 1, n_mfcc, frames) — single channel
        Conv2d(1, 16, 3×3) → BN → ReLU → MaxPool2d(2)
        Conv2d(16, 32, 3×3) → BN → ReLU → MaxPool2d(2)
        AdaptiveAvgPool2d → (4, 8) spatial
        Flatten → Dropout(0.3) → Linear → Logits

    Designed for quick pipeline validation with small input sizes
    (e.g., n_mfcc=20, frames=64). Not intended as a final model.

    Attributes:
        n_mfcc: Number of MFCC coefficient channels in the input.
        n_classes: Number of output genre classes.
    """

    def __init__(
        self,
        n_mfcc: int = 20,
        n_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the debug CNN.

        Args:
            n_mfcc: Number of MFCC channels in the input tensor (H dimension).
            n_classes: Number of genre classes for the output layer.
            dropout: Dropout probability after pooling (default: 0.3).
        """
        super().__init__()

        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
        self.dropout_rate = dropout

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d((4, 8))

        self.fc = nn.Linear(32 * 4 * 8, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        logger.debug(
            "MiniCNN initialized — n_mfcc=%d, n_classes=%d, dropout=%.2f",
            n_mfcc,
            n_classes,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Can be 3D (B, n_mfcc, frames) or 4D
               (B, channels, n_mfcc, frames). If 3D, a channel dimension
               is inserted automatically. If 4D with channels > 1, the
               channel dimension is used as-is (no unsqueeze).

        Returns:
            Logits tensor of shape (B, n_classes).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def save(self, path: Path) -> None:
        """
        Save model state and configuration to a checkpoint file.

        Args:
            path: File path for the saved checkpoint (.pt extension recommended).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "n_mfcc": self.n_mfcc,
            "n_classes": self.n_classes,
            "dropout": self.dropout_rate,
        }
        torch.save(checkpoint, path)
        logger.info("MiniCNN saved: %s", path)

    @classmethod
    def load(
        cls,
        path: Path,
        device: Optional[str] = None,
    ) -> "MiniCNN":
        """
        Load model from a checkpoint file.

        Args:
            path: Path to the saved checkpoint.
            device: Device to load the model onto ('cpu', 'cuda', etc.).
                    Defaults to 'cpu' if None.

        Returns:
            MiniCNN instance with restored weights and configuration.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if device is None:
            device = "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        n_mfcc = checkpoint.get("n_mfcc", 20)
        n_classes = checkpoint.get("n_classes", 10)
        dropout = checkpoint.get("dropout", 0.3)

        model = cls(n_mfcc=n_mfcc, n_classes=n_classes, dropout=dropout)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        logger.info(
            "MiniCNN loaded: %s (n_mfcc=%d, n_classes=%d) → %s",
            path,
            n_mfcc,
            n_classes,
            device,
        )

        return model

    def get_num_parameters(self) -> int:
        """
        Count the total number of trainable parameters.

        Returns:
            Integer parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_info(self) -> None:
        """
        Print a summary of the model architecture and parameter count.

        This is a manual debugging/exploration utility.
        """
        print("=" * 50)
        print("MiniCNN — Debug Model")
        print("=" * 50)
        print(f"Input:        (B, {self.n_mfcc}, frames)")
        print(f"Classes:      {self.n_classes}")
        print(f"Dropout:      {self.dropout_rate}")
        print(f"Parameters:   {self.get_num_parameters():,}")
        print("=" * 50)
        print(self)