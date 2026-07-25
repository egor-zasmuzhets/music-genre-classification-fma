"""
AudioCNN — convolutional neural network for MFCC-based genre classification.

A medium-depth CNN architecture optimized for MFCC inputs as 2D spectrograms.
Input shape: (batch, channels, n_mfcc, time_frames)
- channels: 1 for raw MFCC, 3 for MFCC + delta + delta-delta
- n_mfcc: number of MFCC coefficient bands (frequency axis)
- time_frames: temporal dimension

The architecture preserves the 2D time-frequency structure, using convolutions
to capture local patterns across both frequency and time axes simultaneously.

Design updates (v3):
- Added target_frames parameter for dynamic classifier sizing
- Reduced default dropout (0.3/0.2) for imbalanced datasets
- Adaptive architecture for variable input lengths

Stacking support (v4):
- Added predict_proba() method for ensemble probabilities
- Added export_to_onnx() for production deployment

Typical usage:
    from src.models.cnn_audio import AudioCNN

    model = AudioCNN(n_mfcc=40, n_channels=3, n_classes=16, target_frames=430)
    output = model(mfcc_batch)  # (B, 3, 40, 430) -> (B, 16)

    # For stacking ensemble
    probabilities = model.predict_proba(mfcc_batch)  # (B, n_classes)

    model.save(Path("checkpoints/cnn_model.pt"))
    model = AudioCNN.load(Path("checkpoints/cnn_model.pt"))
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


logger = logging.getLogger(__name__)


class AudioCNN(nn.Module):
    """
    2D CNN for MFCC spectrogram classification with dynamic sizing.

    Architecture:
        Input:  (B, n_channels, n_mfcc, time_frames)

        Block 1: Conv2d(n_channels, 64, 5x5) → BN → ReLU → MaxPool2d(2)
        Block 2: Conv2d(64, 128, 5x5) → BN → ReLU → MaxPool2d(2)
        Block 3: Conv2d(128, 256, 3x3) → BN → ReLU → MaxPool2d(2)

        Classifier:
            Adaptive pooling → Flatten → Dropout → FC(?, 512) → ReLU
            → Dropout → FC(512, 256) → ReLU → Dropout → FC(256, n_classes)

    Attributes:
        n_mfcc: Number of MFCC coefficient bands.
        n_channels: Number of input channels (1 for raw, 3 with deltas).
        n_classes: Number of output genre classes.
        dropout_rate: Dropout probability after convolutional stem.
        fc_dropout_rate: Dropout probability between FC layers.
        target_frames: Expected number of time frames in input.
    """

    def __init__(
        self,
        n_mfcc: int = 40,
        n_channels: int = 3,
        n_classes: int = 16,
        dropout: float = 0.3,
        fc_dropout: float = 0.2,
        target_frames: int = 430,
    ) -> None:
        super().__init__()

        self.n_mfcc = n_mfcc
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_rate = dropout
        self.fc_dropout_rate = fc_dropout
        self.target_frames = target_frames

        self.conv_block1 = self._make_conv_block(
            n_channels, 64, kernel_size=5, pool_size=2
        )
        self.conv_block2 = self._make_conv_block(64, 128, kernel_size=5, pool_size=2)
        self.conv_block3 = self._make_conv_block(128, 256, kernel_size=3, pool_size=2)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self._compute_classifier_input_size(target_frames)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._fc_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(256, n_classes),
        )

        self._initialize_weights()

        total_params = self.get_num_parameters()
        logger.info(
            "AudioCNN v4 initialized — input=(B,%d,%d,%d), classes=%d, "
            "dropout=%.2f/%.2f, params=%d",
            n_channels, n_mfcc, target_frames, n_classes,
            dropout, fc_dropout, total_params
        )

    @staticmethod
    def _make_conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> nn.Sequential:
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def _compute_classifier_input_size(self, target_frames: int) -> None:
        dummy_input = torch.zeros(1, self.n_channels, self.n_mfcc, target_frames)
        with torch.no_grad():
            x = self.conv_block1(dummy_input)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            x = self.conv_block5(x)
            x = self.freq_pool(x)
        self._fc_input_size = x.view(1, -1).size(1)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def set_dropout(self, dropout: float, fc_dropout: float) -> None:
        """Change dropout rates on-the-fly for progressive training."""
        self.dropout_rate = dropout
        self.fc_dropout_rate = fc_dropout

        dropout_layers = [m for m in self.classifier if isinstance(m, nn.Dropout)]
        if len(dropout_layers) >= 1:
            dropout_layers[0].p = dropout
        if len(dropout_layers) >= 2:
            dropout_layers[1].p = fc_dropout
        if len(dropout_layers) >= 3:
            dropout_layers[2].p = fc_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.freq_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Return class probabilities for stacking ensemble.

        Args:
            x: Input tensor. Can be 2D, 3D, or 4D.

        Returns:
            numpy array of shape (B, n_classes) with softmax probabilities.
        """
        self.eval()

        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        logits = self.forward(x)
        proba = torch.softmax(logits, dim=1)
        return proba.cpu().numpy()

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> np.ndarray:
        """
        Return raw logits (before softmax).

        Args:
            x: Input tensor.

        Returns:
            numpy array of shape (B, n_classes) with logits.
        """
        self.eval()

        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        return self.forward(x).cpu().numpy()

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract embeddings before the final classifier layer.

        Useful for hierarchical ensemble approaches.

        Args:
            x: Input tensor.

        Returns:
            numpy array of shape (B, embedding_dim) where embedding_dim is 1024.
        """
        self.eval()

        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.freq_pool(x)
        x = x.view(x.size(0), -1)

        return x.cpu().numpy()

    def export_to_onnx(
        self,
        save_path: Path,
        dummy_input_shape: tuple = (1, 3, 40, 430),
        opset_version: int = 14
    ) -> None:
        """
        Export model to ONNX format for production inference.

        Args:
            save_path: Path to save the .onnx file.
            dummy_input_shape: Shape for tracing (batch, channels, mfcc, frames).
            opset_version: ONNX opset version (default 14).
        """
        self.eval()
        dummy_input = torch.randn(dummy_input_shape)

        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            input_names=['mfcc_input'],
            output_names=['logits', 'proba'],
            dynamic_axes={
                'mfcc_input': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'proba': {0: 'batch_size'}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        logger.info(f"CNN exported to ONNX: {save_path}")

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Path) -> None:
        """Save model state and configuration to a checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "n_mfcc": self.n_mfcc,
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "dropout": self.dropout_rate,
            "fc_dropout": self.fc_dropout_rate,
            "target_frames": self.target_frames,
        }
        torch.save(checkpoint, path)
        logger.info("AudioCNN saved: %s (params=%d)", path, self.get_num_parameters())

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> "AudioCNN":
        """Load model from a checkpoint file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if device is None:
            device = "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = cls(
            n_mfcc=checkpoint.get("n_mfcc", 40),
            n_channels=checkpoint.get("n_channels", 3),
            n_classes=checkpoint.get("n_classes", 16),
            dropout=checkpoint.get("dropout", 0.3),
            fc_dropout=checkpoint.get("fc_dropout", 0.2),
            target_frames=checkpoint.get("target_frames", 430),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        logger.info(
            "AudioCNN loaded: %s (channels=%d, mfcc=%d, classes=%d, frames=%d) → %s",
            path.name, model.n_channels, model.n_mfcc,
            model.n_classes, model.target_frames, device
        )

        return model

    def print_info(self) -> None:
        """Print a human-readable summary of the model architecture."""
        print("=" * 70)
        print("AudioCNN v4 — 2D CNN for MFCC Spectrograms with Stacking Support")
        print("=" * 70)
        print(f"Input shape:      (B, {self.n_channels}, {self.n_mfcc}, {self.target_frames})")
        print(f"Output classes:   {self.n_classes}")
        print(f"Dropout rates:    {self.dropout_rate} / {self.fc_dropout_rate}")
        print(f"Parameters:       {self.get_num_parameters():,}")
        print(f"Classifier input: {self._fc_input_size:,}")
        print("")
        print("Methods for stacking:")
        print("  - predict_proba()  → probabilities for ensemble")
        print("  - predict_logits() → raw logits")
        print("  - get_embeddings() → features before classifier")
        print("  - export_to_onnx() → production deployment")
        print("=" * 70)