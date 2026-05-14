"""
src/data/torch_dataset.py
PyTorch Dataset for MFCC-based audio classification with augmentation.

Loads MFCC features for tracks identified by pipeline indices,
applies optional SpecAugment-style augmentation, and converts
to PyTorch tensors in CNN-compatible format. Gracefully handles
corrupt audio by returning zero-filled tensors and tracking failures.

Typical usage:
    from src.data.torch_dataset import create_mfcc_dataloaders

    train_loader, val_loader, test_loader, genres, report = \
        create_mfcc_dataloaders(
            subset="medium",
            batch_size=32,
            target_frames=128,
            augment_train=True,
        )

    for mfcc, labels in train_loader:
        # mfcc: (batch, n_channels, 1, target_frames)
        # labels: (batch,)
        ...
"""

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.audio_loader import AudioLoader
from src.data.load_processed import load_data, load_track_indices
from src.data.mfcc_extractor import MFCCConfig, MFCCExtractor


logger = logging.getLogger(__name__)


class MFCCAugmentation:
    """
    SpecAugment-style data augmentation for MFCC matrices.

    Applies time masking, frequency masking, and additive Gaussian noise
    with configurable probabilities and magnitudes.

    All methods are static — no internal state is maintained.
    """

    @staticmethod
    def time_masking(
        mfcc: np.ndarray,
        max_mask_ratio: float = 0.1,
    ) -> np.ndarray:
        """
        Zero out a random contiguous segment along the time axis.

        Args:
            mfcc: Array of shape (channels, frames) or (channels, height, frames).
            max_mask_ratio: Maximum fraction of the time axis to mask.

        Returns:
            Augmented array (copy).
        """
        if mfcc.ndim == 2:
            _, width = mfcc.shape
            mask_len = int(width * random.uniform(0.05, max_mask_ratio))
            mask_len = max(1, min(mask_len, width - 2))
            mask_start = (
                random.randint(0, width - mask_len - 1)
                if width > mask_len + 1
                else 0
            )
            mfcc_aug = mfcc.copy()
            mfcc_aug[:, mask_start : mask_start + mask_len] = 0
            return mfcc_aug
        else:
            _, _, width = mfcc.shape
            mask_len = int(width * random.uniform(0.05, max_mask_ratio))
            mask_len = max(1, min(mask_len, width - 2))
            mask_start = (
                random.randint(0, width - mask_len - 1)
                if width > mask_len + 1
                else 0
            )
            mfcc_aug = mfcc.copy()
            mfcc_aug[:, :, mask_start : mask_start + mask_len] = 0
            return mfcc_aug

    @staticmethod
    def frequency_masking(
        mfcc: np.ndarray,
        max_mask_ratio: float = 0.15,
    ) -> np.ndarray:
        """
        Zero out a random contiguous band along the frequency axis.

        Only applies to 3D arrays (channels, height, width).
        For 2D arrays (channels, frames), returns unchanged.

        Args:
            mfcc: Array of shape (channels, height, frames) or (channels, frames).
            max_mask_ratio: Maximum fraction of the frequency axis to mask.

        Returns:
            Augmented array (copy).
        """
        if mfcc.ndim == 2:
            return mfcc

        _, height, _ = mfcc.shape
        mask_len = int(height * random.uniform(0.05, max_mask_ratio))
        mask_len = max(1, min(mask_len, height - 2))
        mask_start = (
            random.randint(0, height - mask_len - 1)
            if height > mask_len + 1
            else 0
        )
        mfcc_aug = mfcc.copy()
        mfcc_aug[:, mask_start : mask_start + mask_len, :] = 0
        return mfcc_aug

    @staticmethod
    def add_noise(
        mfcc: np.ndarray,
        noise_std: float = 0.005,
    ) -> np.ndarray:
        """
        Add small Gaussian noise to the MFCC matrix.

        Args:
            mfcc: Array of any shape.
            noise_std: Standard deviation of the Gaussian noise.

        Returns:
            Array with added noise (not a copy — input is modified).
        """
        noise = np.random.normal(0, noise_std, mfcc.shape)
        return mfcc + noise

    @classmethod
    def apply_spec_augment(
        cls,
        mfcc: np.ndarray,
        time_mask_ratio: float = 0.1,
        freq_mask_ratio: float = 0.15,
        p: float = 0.5,
    ) -> np.ndarray:
        """
        Apply SpecAugment with a given probability.

        With probability p, applies time masking and/or frequency masking
        (each independently at 50% chance). Only frequency masking is
        applied to 3D inputs.

        Args:
            mfcc: MFCC array, (channels, frames) or (channels, height, frames).
            time_mask_ratio: Max fraction of time axis to mask.
            freq_mask_ratio: Max fraction of frequency axis to mask.
            p: Overall probability of applying any augmentation.

        Returns:
            Augmented array (copy).
        """
        if random.random() > p:
            return mfcc

        mfcc_aug = mfcc.copy()

        if random.random() < 0.5:
            mfcc_aug = cls.time_masking(mfcc_aug, time_mask_ratio)

        if mfcc_aug.ndim == 3 and random.random() < 0.5:
            mfcc_aug = cls.frequency_masking(mfcc_aug, freq_mask_ratio)

        return mfcc_aug


class MFCCDataset(Dataset):
    """
    PyTorch Dataset that loads MFCC features for CNN training.

    Maps track indices from the data pipeline to MFCC matrices via
    MFCCExtractor. Supports in-memory caching, optional SpecAugment
    augmentation (train only), and graceful handling of failed tracks.

    Failed tracks produce zero-filled tensors of the correct shape,
    and are recorded for later analysis via get_failed_tracks_report().
    The underlying MFCCExtractor caches failures, so repeated access
    to a corrupt track during multi-epoch training is instantaneous.

    Attributes:
        indices: List of track IDs in this dataset split.
        labels: Corresponding integer label for each track.
        target_frames: Number of time frames per sample (axis=-1).
        augment: Whether to apply augmentation.
        augmentation_p: Probability of augmentation.
        failed_tracks: List of track IDs that failed to load.
    """

    def __init__(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        audio_loader: Optional[AudioLoader] = None,
        mfcc_config: Optional[MFCCConfig] = None,
        target_frames: int = 128,
        use_deltas: bool = True,
        cache_mfcc: bool = True,
        use_disk_cache: bool = True,
        augment: bool = False,
        augmentation_p: float = 0.5,
        track_failure_handler: Optional[Callable] = None,
    ):
        """
        Initialize the MFCC dataset.

        Args:
            indices: Array of track IDs for this split.
            labels: Array of integer class labels (aligned with indices).
            audio_loader: AudioLoader instance. Created if None.
            mfcc_config: MFCC extraction config. Default if None.
            target_frames: Exact number of time frames per sample.
            use_deltas: Include delta and delta-delta coefficients.
            cache_mfcc: Cache extracted MFCCs in memory.
            use_disk_cache: Cache extracted MFCCs on disk.
            augment: Apply SpecAugment (typically True for train only).
            augmentation_p: Probability of applying augmentation per sample.
            track_failure_handler: Optional callback(track_id, status)
                                   invoked when a track fails to load.
        """
        self.indices = (
            indices.tolist() if hasattr(indices, "tolist") else list(indices)
        )
        self.labels = (
            labels.tolist() if hasattr(labels, "tolist") else list(labels)
        )
        self.target_frames = target_frames
        self.use_deltas = use_deltas
        self.cache_mfcc = cache_mfcc
        self.augment = augment
        self.augmentation_p = augmentation_p
        self.track_failure_handler = track_failure_handler

        self.failed_tracks: List[int] = []
        self.failed_tracks_details: Dict[int, Dict[str, Any]] = {}

        if mfcc_config is None:
            mfcc_config = MFCCConfig(
                include_delta=use_deltas,
                include_delta2=use_deltas,
            )

        self.extractor = MFCCExtractor(
            config=mfcc_config,
            audio_loader=audio_loader or AudioLoader(),
            use_disk_cache=use_disk_cache,
        )

        self._cache: Dict[int, np.ndarray] = {}

        n_channels = 1
        if use_deltas:
            n_channels += 1
            n_channels += 1

        logger.info(
            "MFCCDataset created — %d samples, target_frames=%d, "
            "deltas=%s, channels=%d, augment=%s (p=%.2f), "
            "memory_cache=%s, disk_cache=%s",
            len(self.indices),
            target_frames,
            "on" if use_deltas else "off",
            n_channels,
            "on" if augment else "off",
            augmentation_p,
            "on" if cache_mfcc else "off",
            "on" if use_disk_cache else "off",
        )

    def __len__(self) -> int:
        """Return the number of samples in this dataset split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample by index.

        Args:
            idx: Sample index (0 to len-1).

        Returns:
            Tuple of (mfcc_tensor, label_tensor):
            - mfcc_tensor: (n_channels, 1, target_frames) float32
            - label_tensor: scalar int64
        """
        track_id = self.indices[idx]
        label = self.labels[idx]

        if self.cache_mfcc and track_id in self._cache:
            mfcc = self._cache[track_id]
        else:
            mfcc, status = self.extractor.prepare_for_cnn_with_status(
                track_id, self.target_frames
            )

            if mfcc is None:
                self._record_failure(track_id, status)
                n_channels = self._get_n_channels()
                mfcc = np.zeros(
                    (n_channels, self.target_frames), dtype=np.float32
                )
            elif self.cache_mfcc:
                self._cache[track_id] = mfcc

        if self.augment:
            mfcc = MFCCAugmentation.apply_spec_augment(
                mfcc, p=self.augmentation_p
            )
            if random.random() < 0.3:
                mfcc = MFCCAugmentation.add_noise(mfcc, noise_std=0.003)

        if mfcc.ndim == 2:
            mfcc = mfcc[:, np.newaxis, :]

        mfcc_tensor = torch.from_numpy(mfcc.copy()).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mfcc_tensor, label_tensor

    def _get_n_channels(self) -> int:
        """Calculate the number of MFCC channels from config."""
        n = self.extractor.config.n_mfcc
        if self.extractor.config.include_delta:
            n += n
        if self.extractor.config.include_delta2:
            n += n
        return n

    def _record_failure(self, track_id: int, status: Dict[str, Any]) -> None:
        """
        Record a failed track and invoke the failure handler if set.

        Tracks already known to be bad are not re-recorded and do not
        trigger the handler again.

        Args:
            track_id: Track that failed.
            status: Status dictionary from the extractor.
        """
        if track_id in self.failed_tracks:
            return

        self.failed_tracks.append(track_id)
        self.failed_tracks_details[track_id] = {
            "error_type": status.get("error_type", "unknown"),
            "error_message": status.get("error_message", ""),
            "split": "train" if self.augment else "val/test",
        }

        if self.track_failure_handler:
            try:
                self.track_failure_handler(track_id, status)
            except Exception:
                logger.warning(
                    "Failure handler raised for track %d", track_id
                )

        logger.debug(
            "Track %d failed — type=%s, split=%s",
            track_id,
            status.get("error_type", "unknown"),
            "train" if self.augment else "val/test",
        )

    def get_failed_tracks_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of failed track loads.

        Returns:
            Dictionary with keys: failed_count, failed_tracks,
            failed_details, total_tracks, success_rate.
        """
        n_total = len(self.indices)
        n_failed = len(self.failed_tracks)
        return {
            "failed_count": n_failed,
            "failed_tracks": self.failed_tracks,
            "failed_details": self.failed_tracks_details,
            "total_tracks": n_total,
            "success_rate": (n_total - n_failed) / n_total if n_total > 0 else 0.0,
        }

    def get_successful_indices(self) -> List[int]:
        """
        Return track IDs that were loaded successfully so far.

        Note: only reflects tracks already accessed via __getitem__.
        """
        return [idx for idx in self.indices if idx not in self.failed_tracks]

    def get_failed_indices(self) -> List[int]:
        """Return track IDs that failed to load."""
        return self.failed_tracks.copy()

    def clear_failed_tracks(self) -> None:
        """Reset the failed track records."""
        self.failed_tracks.clear()
        self.failed_tracks_details.clear()

    def get_cached_count(self) -> int:
        """Return the number of samples currently in the memory cache."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the in-memory MFCC cache."""
        n_cleared = len(self._cache)
        self._cache.clear()
        logger.debug("Memory cache cleared: %d entries removed", n_cleared)


def create_mfcc_dataloaders(
    subset: str = "medium",
    min_samples_per_genre: int = 100,
    dataset_id: Optional[str] = None,
    batch_size: int = 32,
    target_frames: int = 128,
    use_deltas: bool = True,
    num_workers: int = 0,
    cache_mfcc: bool = True,
    use_disk_cache: bool = True,
    augment_train: bool = True,
    failure_handler: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, Any]]:
    """
    Create train/val/test DataLoaders for MFCC-based classification.

    Loads preprocessed indices and labels from the data pipeline cache,
    builds MFCCDataset instances for each split, and wraps them in
    DataLoaders with appropriate shuffle settings.

    Args:
        subset: FMA subset ('small', 'medium', 'large').
        min_samples_per_genre: Minimum tracks per genre used in pipeline.
        dataset_id: Full dataset ID string (overrides subset/min if provided).
        batch_size: Samples per batch.
        target_frames: Number of time frames per MFCC sample.
        use_deltas: Include delta and delta-delta coefficients.
        num_workers: Number of DataLoader worker processes.
        cache_mfcc: Cache extracted MFCCs in memory.
        use_disk_cache: Cache extracted MFCCs on disk.
        augment_train: Apply SpecAugment to the training set.
        failure_handler: Callback for individual track load failures.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, genre_names, failure_report).
        failure_report: dict with keys 'train', 'val', 'test', each containing
                        a per-split failure summary from get_failed_tracks_report().
    """
    data = load_data(
        subset=subset,
        min_samples_per_genre=min_samples_per_genre,
        dataset_id=dataset_id,
    )
    train_indices, val_indices, test_indices = load_track_indices(
        subset=subset,
        min_samples_per_genre=min_samples_per_genre,
        dataset_id=dataset_id,
    )

    genre_names: List[str] = data["genre_names"]

    train_idx_list = (
        train_indices.tolist()
        if hasattr(train_indices, "tolist")
        else list(train_indices)
    )
    val_idx_list = (
        val_indices.tolist()
        if hasattr(val_indices, "tolist")
        else list(val_indices)
    )
    test_idx_list = (
        test_indices.tolist()
        if hasattr(test_indices, "tolist")
        else list(test_indices)
    )

    y_train_list = (
        data["y_train"].tolist()
        if hasattr(data["y_train"], "tolist")
        else list(data["y_train"])
    )
    y_val_list = (
        data["y_val"].tolist()
        if hasattr(data["y_val"], "tolist")
        else list(data["y_val"])
    )
    y_test_list = (
        data["y_test"].tolist()
        if hasattr(data["y_test"], "tolist")
        else list(data["y_test"])
    )

    assert len(train_idx_list) == len(y_train_list), (
        f"Train index/label mismatch: {len(train_idx_list)} vs {len(y_train_list)}"
    )
    assert len(val_idx_list) == len(y_val_list), (
        f"Val index/label mismatch: {len(val_idx_list)} vs {len(y_val_list)}"
    )
    assert len(test_idx_list) == len(y_test_list), (
        f"Test index/label mismatch: {len(test_idx_list)} vs {len(y_test_list)}"
    )

    logger.info(
        "Creating MFCC DataLoaders — subset=%s, %d train / %d val / %d test, "
        "%d genres, batch_size=%d, target_frames=%d, augment=%s",
        subset.upper(),
        len(train_idx_list),
        len(val_idx_list),
        len(test_idx_list),
        len(genre_names),
        batch_size,
        target_frames,
        "on" if augment_train else "off",
    )

    shared_loader = AudioLoader()

    train_dataset = MFCCDataset(
        indices=np.array(train_idx_list),
        labels=np.array(y_train_list),
        audio_loader=shared_loader,
        target_frames=target_frames,
        use_deltas=use_deltas,
        cache_mfcc=cache_mfcc,
        use_disk_cache=use_disk_cache,
        augment=augment_train,
        augmentation_p=0.5,
        track_failure_handler=failure_handler,
    )

    val_dataset = MFCCDataset(
        indices=np.array(val_idx_list),
        labels=np.array(y_val_list),
        audio_loader=shared_loader,
        target_frames=target_frames,
        use_deltas=use_deltas,
        cache_mfcc=cache_mfcc,
        use_disk_cache=use_disk_cache,
        augment=False,
        augmentation_p=0.0,
        track_failure_handler=failure_handler,
    )

    test_dataset = MFCCDataset(
        indices=np.array(test_idx_list),
        labels=np.array(y_test_list),
        audio_loader=shared_loader,
        target_frames=target_frames,
        use_deltas=use_deltas,
        cache_mfcc=cache_mfcc,
        use_disk_cache=use_disk_cache,
        augment=False,
        augmentation_p=0.0,
        track_failure_handler=failure_handler,
    )

    failure_report = {
        "train": train_dataset.get_failed_tracks_report(),
        "val": val_dataset.get_failed_tracks_report(),
        "test": test_dataset.get_failed_tracks_report(),
    }

    total_failed = (
        failure_report["train"]["failed_count"]
        + failure_report["val"]["failed_count"]
        + failure_report["test"]["failed_count"]
    )

    if total_failed > 0:
        logger.warning(
            "Total failed tracks across splits: %d "
            "(train: %d, val: %d, test: %d)",
            total_failed,
            failure_report["train"]["failed_count"],
            failure_report["val"]["failed_count"],
            failure_report["test"]["failed_count"],
        )
    else:
        logger.info("All tracks loaded successfully across all splits")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "DataLoaders created — train: %d batches, val: %d, test: %d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    return train_loader, val_loader, test_loader, genre_names, failure_report


if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    setup_logging(level=logging.DEBUG, mode="console")

    train_loader, val_loader, test_loader, genres, report = \
        create_mfcc_dataloaders(
            subset="medium",
            min_samples_per_genre=10,
            batch_size=4,
            target_frames=64,
            use_deltas=False,
            augment_train=False,
        )

    print(f"\nFailure report:")
    for split, r in report.items():
        print(f"  {split}: {r['failed_count']}/{r['total_tracks']} "
              f"({r['success_rate']:.2%})")

    if report["train"]["failed_count"] > 0:
        print(f"\nFailed track IDs (first 10): "
              f"{report['train']['failed_tracks'][:10]}")

    for mfcc, labels in train_loader:
        print(f"\nSample batch — MFCC shape: {mfcc.shape}, labels: {labels}")
        break