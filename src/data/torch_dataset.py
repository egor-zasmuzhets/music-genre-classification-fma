"""
PyTorch Dataset for MFCC-based audio classification with augmentation.

Loads MFCC features in correct 2D spectrogram format (channels, n_mfcc, frames)
for CNN processing. Applies class-aware augmentation with stronger transforms
for rare classes, mixup within same class, and balanced sampling.

Supports curriculum augmentation — augmentation strength grows with epoch.
"""

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.audio_loader import AudioLoader
from src.data.load_processed import load_data, load_track_indices
from src.data.mfcc_extractor import MFCCConfig, MFCCExtractor


logger = logging.getLogger(__name__)


class MFCCAugmentation:
    """
    Class-aware data augmentation for MFCC spectrograms.

    Supports curriculum learning: augmentation strength grows with epoch.
    frequency_masking is REMOVED — replaced by frequency_warp.
    """

    # ========================================================================
    # MASKING
    # ========================================================================

    @staticmethod
    def time_masking(mfcc: np.ndarray, max_mask_ratio: float = 0.1) -> np.ndarray:
        """Zero out a random contiguous segment along the time axis."""
        if mfcc.ndim == 3:
            _, _, width = mfcc.shape
        else:
            _, width = mfcc.shape
        mask_len = int(width * random.uniform(0.05, max_mask_ratio))
        mask_len = max(1, min(mask_len, width - 2))
        mask_start = random.randint(0, width - mask_len - 1) if width > mask_len + 1 else 0
        mfcc_aug = mfcc.copy()
        if mfcc.ndim == 3:
            mfcc_aug[:, :, mask_start:mask_start + mask_len] = 0
        else:
            mfcc_aug[:, mask_start:mask_start + mask_len] = 0
        return mfcc_aug

    # ========================================================================
    # TEMPORAL TRANSFORMS
    # ========================================================================

    @staticmethod
    def time_roll(mfcc: np.ndarray) -> np.ndarray:
        """Cyclic shift along the time axis."""
        if mfcc.ndim == 3:
            shift = random.randint(0, mfcc.shape[2] - 1)
            return np.roll(mfcc, shift, axis=2)
        elif mfcc.ndim == 2:
            shift = random.randint(0, mfcc.shape[1] - 1)
            return np.roll(mfcc, shift, axis=1)
        return mfcc

    @staticmethod
    def time_stretch(mfcc: np.ndarray, rate_range: tuple = (0.92, 1.08)) -> np.ndarray:
        """Stretch/compress MFCC along the time axis via interpolation."""
        rate = random.uniform(*rate_range)
        _, n_mfcc, n_frames = mfcc.shape
        new_width = int(n_frames * rate)
        stretched = np.zeros((mfcc.shape[0], n_mfcc, max(new_width, n_frames)), dtype=mfcc.dtype)
        x_old = np.linspace(0, n_frames - 1, n_frames)
        x_new = np.linspace(0, n_frames - 1, new_width)
        for c in range(mfcc.shape[0]):
            for h in range(n_mfcc):
                stretched[c, h, :new_width] = np.interp(x_new, x_old, mfcc[c, h])
        if new_width > n_frames:
            start = (new_width - n_frames) // 2
            return stretched[:, :, start:start + n_frames].astype(np.float32)
        else:
            return stretched[:, :, :n_frames].astype(np.float32)

    # ========================================================================
    # FREQUENCY TRANSFORMS
    # ========================================================================

    @staticmethod
    def pitch_shift_mfcc(mfcc: np.ndarray, n_steps: int = 3) -> np.ndarray:
        """Shift MFCC up/down along the frequency axis."""
        shift = random.randint(-n_steps, n_steps)
        if shift == 0:
            return mfcc
        shifted = np.zeros_like(mfcc)
        if shift > 0:
            shifted[:, shift:, :] = mfcc[:, :-shift, :]
        else:
            shift = abs(shift)
            shifted[:, :-shift, :] = mfcc[:, shift:, :]
        return shifted

    @staticmethod
    def random_eq(mfcc: np.ndarray, n_bands: int = 4, gain_range: float = 0.25) -> np.ndarray:
        """Apply random equalizer."""
        augmented = mfcc.copy()
        n_mfcc = mfcc.shape[1]
        freqs = np.arange(n_mfcc)
        for c in range(mfcc.shape[0]):
            eq_curve = np.ones(n_mfcc)
            for _ in range(n_bands):
                center = random.randint(0, n_mfcc - 1)
                width = max(1, random.randint(2, n_mfcc // 4))
                gain = 1.0 + random.uniform(-gain_range, gain_range)
                window = np.exp(-0.5 * ((freqs - center) / width) ** 2)
                eq_curve *= (1.0 + (gain - 1.0) * window)
            augmented[c] *= eq_curve[:, np.newaxis]
        return augmented

    @staticmethod
    def harmonic_emphasis(mfcc: np.ndarray, strength: float = 0.25) -> np.ndarray:
        """Vary balance between harmonic and percussive components."""
        s = random.uniform(-strength, strength)
        augmented = mfcc.copy()
        augmented[:, :20, :] *= (1.0 + s)
        augmented[:, 20:, :] *= (1.0 - s)
        return augmented

    @staticmethod
    def frequency_warp(mfcc: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Non-linearly warp the frequency axis."""
        n_mfcc = mfcc.shape[1]
        src = np.arange(n_mfcc)
        warp = strength * np.sin(np.linspace(0, np.pi, n_mfcc))
        dst = src + warp * n_mfcc / 4
        dst = np.clip(dst, 0, n_mfcc - 1)
        warped = np.zeros_like(mfcc)
        for c in range(mfcc.shape[0]):
            for t in range(mfcc.shape[2]):
                warped[c, :, t] = np.interp(src, dst, mfcc[c, :, t])
        return warped

    # ========================================================================
    # NOISE & SPACE
    # ========================================================================

    @staticmethod
    def add_noise(mfcc: np.ndarray, noise_std: float = 0.005) -> np.ndarray:
        """Add white Gaussian noise."""
        noise = np.random.normal(0, noise_std, mfcc.shape)
        return mfcc + noise

    @staticmethod
    def colored_noise(mfcc: np.ndarray, strength: float = 0.005) -> np.ndarray:
        """Add frequency-dependent noise."""
        n_mfcc = mfcc.shape[1]
        noise_profile = np.linspace(0.5, 1.5, n_mfcc)
        noise = np.random.normal(0, strength, mfcc.shape)
        noise *= noise_profile[np.newaxis, :, np.newaxis]
        return mfcc + noise

    @staticmethod
    def reverb_sim(mfcc: np.ndarray, decay: float = 0.2) -> np.ndarray:
        """Simulate reverberation."""
        d = random.uniform(0.1, decay)
        augmented = mfcc.copy()
        impulse = np.exp(-d * np.arange(mfcc.shape[2]))
        impulse /= impulse.sum()
        for c in range(mfcc.shape[0]):
            for h in range(mfcc.shape[1]):
                augmented[c, h] = np.convolve(mfcc[c, h], impulse, mode='same')
        return augmented

    # ========================================================================
    # MIXUP
    # ========================================================================

    @staticmethod
    def mixup_same_class(mfcc: np.ndarray, other_mfcc: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Mix two MFCC samples from the same class."""
        lam = np.random.beta(alpha, alpha)
        return lam * mfcc + (1.0 - lam) * other_mfcc

    # ========================================================================
    # CURRICULUM AUGMENTATION
    # ========================================================================

    @classmethod
    def apply_curriculum_augmentation(
        cls,
        mfcc: np.ndarray,
        class_rarity: str = "common",
        epoch: int = 1,
        total_epochs: int = 8,
    ) -> np.ndarray:
        """
        Augmentation strength grows with epoch (curriculum learning).

        Args:
            mfcc: MFCC array, shape (n_channels, n_mfcc, target_frames).
            class_rarity: 'very_rare', 'rare', 'medium', or 'common'.
            epoch: Current epoch (1-based).
            total_epochs: Total epochs in current stage.

        Returns:
            Augmented array.
        """
        # Progress: 0.3 → 1.0 over the stage
        progress = 0.3 + 0.7 * (epoch - 1) / max(total_epochs - 1, 1)

        if class_rarity == "very_rare":
            aug_pipeline = [
                ("time_stretch", 0.4 + 0.2*progress, {"rate_range": (0.92 - 0.02*progress, 1.08 + 0.02*progress)}),
                ("pitch_shift", 0.3 + 0.2*progress, {"n_steps": int(2 + 2*progress)}),
                ("random_eq", 0.4 + 0.2*progress, {"n_bands": 4, "gain_range": 0.15 + 0.15*progress}),
                ("harmonic_emphasis", 0.2 + 0.2*progress, {"strength": 0.15 + 0.15*progress}),
                ("time_roll", 0.3 + 0.2*progress, {}),
                ("reverb_sim", 0.2 + 0.2*progress, {"decay": 0.10 + 0.15*progress}),
                ("colored_noise", 0.2 + 0.2*progress, {"strength": 0.003 + 0.003*progress}),
                ("frequency_warp", 0.2 + 0.2*progress, {"strength": 0.08 + 0.07*progress}),
                ("add_noise", 0.2 + 0.1*progress, {"noise_std": 0.002 + 0.002*progress}),
                ("time_mask", 0.4 + 0.1*progress, {"max_mask_ratio": 0.06 + 0.04*progress}),
            ]
        elif class_rarity == "rare":
            aug_pipeline = [
                ("time_stretch", 0.3 + 0.2*progress, {"rate_range": (0.94 - 0.02*progress, 1.06 + 0.02*progress)}),
                ("pitch_shift", 0.3 + 0.1*progress, {"n_steps": int(1 + 2*progress)}),
                ("random_eq", 0.3 + 0.2*progress, {"n_bands": 4, "gain_range": 0.12 + 0.13*progress}),
                ("harmonic_emphasis", 0.2 + 0.1*progress, {"strength": 0.12 + 0.13*progress}),
                ("time_roll", 0.3 + 0.1*progress, {}),
                ("reverb_sim", 0.2 + 0.1*progress, {"decay": 0.08 + 0.12*progress}),
                ("colored_noise", 0.2 + 0.1*progress, {"strength": 0.002 + 0.002*progress}),
                ("frequency_warp", 0.2 + 0.1*progress, {"strength": 0.06 + 0.06*progress}),
                ("add_noise", 0.2 + 0.1*progress, {"noise_std": 0.002 + 0.001*progress}),
                ("time_mask", 0.5 + 0.1*progress, {"max_mask_ratio": 0.08 + 0.04*progress}),
            ]
        elif class_rarity == "medium":
            aug_pipeline = [
                ("time_stretch", 0.2 + 0.1*progress, {"rate_range": (0.96 - 0.01*progress, 1.04 + 0.01*progress)}),
                ("pitch_shift", 0.2 + 0.1*progress, {"n_steps": int(1 + 1*progress)}),
                ("random_eq", 0.2 + 0.1*progress, {"n_bands": 3, "gain_range": 0.10 + 0.10*progress}),
                ("time_roll", 0.2 + 0.1*progress, {}),
                ("colored_noise", 0.1 + 0.1*progress, {"strength": 0.002 + 0.001*progress}),
                ("frequency_warp", 0.1 + 0.1*progress, {"strength": 0.04 + 0.04*progress}),
                ("add_noise", 0.1 + 0.1*progress, {"noise_std": 0.002 + 0.002*progress}),
                ("time_mask", 0.4 + 0.1*progress, {"max_mask_ratio": 0.06 + 0.04*progress}),
            ]
        else:  # common
            aug_pipeline = [
                ("time_stretch", 0.1 + 0.1*progress, {"rate_range": (0.96, 1.04)}),
                ("random_eq", 0.1 + 0.1*progress, {"n_bands": 2, "gain_range": 0.08 + 0.07*progress}),
                ("time_roll", 0.1 + 0.05*progress, {}),
                ("frequency_warp", 0.1 + 0.05*progress, {"strength": 0.03 + 0.02*progress}),
                ("add_noise", 0.1 + 0.05*progress, {"noise_std": 0.002 + 0.001*progress}),
                ("time_mask", 0.2 + 0.1*progress, {"max_mask_ratio": 0.05 + 0.03*progress}),
            ]

        mfcc_aug = mfcc.copy()

        method_map = {
            "time_mask": cls.time_masking,
            "time_stretch": cls.time_stretch,
            "pitch_shift": cls.pitch_shift_mfcc,
            "random_eq": cls.random_eq,
            "harmonic_emphasis": cls.harmonic_emphasis,
            "time_roll": cls.time_roll,
            "reverb_sim": cls.reverb_sim,
            "colored_noise": cls.colored_noise,
            "frequency_warp": cls.frequency_warp,
            "add_noise": cls.add_noise,
        }

        for aug_name, prob, kwargs in aug_pipeline:
            if random.random() < prob:
                method = method_map.get(aug_name)
                if method:
                    mfcc_aug = method(mfcc_aug, **kwargs) if kwargs else method(mfcc_aug)

        return mfcc_aug


class MFCCDataset(Dataset):
    """
    PyTorch Dataset that loads MFCC features with class-aware sampling.

    Supports curriculum augmentation via set_epoch().
    """

    def __init__(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        audio_loader: Optional[AudioLoader] = None,
        mfcc_config: Optional[MFCCConfig] = None,
        target_frames: int = 430,
        cache_mfcc: bool = True,
        use_disk_cache: bool = True,
        augment: bool = False,
        class_counts: Optional[Dict[int, int]] = None,
        very_rare_threshold: int = 100,
        rare_threshold: int = 500,
        medium_threshold: int = 1400,
        track_failure_handler: Optional[Callable] = None,
    ) -> None:
        self.indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        self.labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        self.target_frames = target_frames
        self.cache_mfcc = cache_mfcc
        self.augment = augment
        self.track_failure_handler = track_failure_handler

        self.failed_tracks: List[int] = []
        self.failed_tracks_details: Dict[int, Dict[str, Any]] = {}

        if mfcc_config is None:
            mfcc_config = MFCCConfig(include_delta=True, include_delta2=True)

        self.n_mfcc = mfcc_config.n_mfcc
        self.n_channels = mfcc_config.get_total_channels()
        self.include_delta = mfcc_config.include_delta
        self.include_delta2 = mfcc_config.include_delta2

        self.extractor = MFCCExtractor(
            config=mfcc_config,
            audio_loader=audio_loader or AudioLoader(),
            use_disk_cache=use_disk_cache,
        )

        self._cache: Dict[Tuple[int, int, int, str], np.ndarray] = {}

        # Class rarity
        self.class_rarity_map: Dict[int, str] = {}
        if class_counts is not None:
            for class_idx, count in class_counts.items():
                if count <= very_rare_threshold:
                    self.class_rarity_map[class_idx] = "very_rare"
                elif count <= rare_threshold:
                    self.class_rarity_map[class_idx] = "rare"
                elif count <= medium_threshold:
                    self.class_rarity_map[class_idx] = "medium"
                else:
                    self.class_rarity_map[class_idx] = "common"
        else:
            for class_idx in set(self.labels):
                self.class_rarity_map[class_idx] = "common"

        self._class_to_indices: Dict[int, List[int]] = {}
        for i, label in enumerate(self.labels):
            if label not in self._class_to_indices:
                self._class_to_indices[label] = []
            self._class_to_indices[label].append(i)

        # Curriculum state
        self.current_epoch: int = 1
        self.total_epochs: int = 25

        vr = sum(1 for v in self.class_rarity_map.values() if v == "very_rare")
        r = sum(1 for v in self.class_rarity_map.values() if v == "rare")
        m = sum(1 for v in self.class_rarity_map.values() if v == "medium")
        c = sum(1 for v in self.class_rarity_map.values() if v == "common")

        logger.info(
            "MFCCDataset created — samples=%d, shape=(%d,%d,%d), augment=%s, "
            "very_rare(≤%d):%d, rare(≤%d):%d, medium(≤%d):%d, common:%d",
            len(self.indices), self.n_channels, self.n_mfcc, target_frames,
            "on" if augment else "off",
            very_rare_threshold, vr, rare_threshold, r, medium_threshold, m, c,
        )

    def set_epoch(self, epoch: int, total_epochs: int = 25) -> None:
        """Update curriculum state for progressive augmentation."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def _get_class_rarity(self, label: int) -> str:
        return self.class_rarity_map.get(label, "common")

    def _get_random_same_class(self, label: int, exclude_idx: int) -> Optional[int]:
        candidates = self._class_to_indices.get(label, [])
        others = [i for i in candidates if i != exclude_idx]
        return random.choice(others) if others else None

    def _get_n_crops(self, class_rarity: str) -> int:
        if class_rarity == "very_rare":
            return 4
        elif class_rarity == "rare":
            return 3
        elif class_rarity == "medium":
            return 2
        else:
            return 1

    def _compute_deltas_on_the_fly(self, mfcc_base: np.ndarray) -> np.ndarray:
        components = [mfcc_base]
        if self.include_delta:
            delta = np.diff(mfcc_base, axis=1, prepend=mfcc_base[:, :1])
            components.append(delta)
        if self.include_delta2:
            if self.include_delta:
                delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
            else:
                delta_first = np.diff(mfcc_base, axis=1, prepend=mfcc_base[:, :1])
                delta2 = np.diff(delta_first, axis=1, prepend=delta_first[:, :1])
            components.append(delta2)
        result = np.stack(components, axis=0).astype(np.float32)
        mean = result.mean(axis=(1, 2), keepdims=True)
        std = result.std(axis=(1, 2), keepdims=True) + 1e-8
        result = (result - mean) / std
        return result

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        track_id = self.indices[idx]
        label = self.labels[idx]
        class_rarity = self._get_class_rarity(label)

        n_crops = self._get_n_crops(class_rarity) if self.augment else 1
        mode = "train" if self.augment else "eval"
        crop_index = random.randint(0, n_crops - 1) if n_crops > 1 else 0
        cache_key = (track_id, self.target_frames, crop_index, mode)

        if self.cache_mfcc and cache_key in self._cache:
            mfcc_base = self._cache[cache_key]
        else:
            crops = self.extractor.prepare_crops_for_track(
                track_id, target_frames=self.target_frames,
                n_crops=n_crops, mode=mode,
            )
            idx_in_crops = min(crop_index, len(crops) - 1) if crops else 0
            mfcc_base = crops[idx_in_crops] if crops and crops[idx_in_crops] is not None else None

            if mfcc_base is None:
                self._record_failure(track_id, {
                    "error_type": "load_failed",
                    "error_message": "MFCC extraction failed",
                })
                mfcc_base = np.zeros((self.n_mfcc, self.target_frames), dtype=np.float32)
            elif self.cache_mfcc:
                self._cache[cache_key] = mfcc_base

        mfcc = self._compute_deltas_on_the_fly(mfcc_base)

        expected_shape = (self.n_channels, self.n_mfcc, self.target_frames)
        if mfcc.shape != expected_shape:
            logger.warning("Track %d has unexpected shape %s, expected %s. Reshaping.",
                           track_id, mfcc.shape, expected_shape)
            if mfcc.size == np.prod(expected_shape):
                mfcc = mfcc.reshape(expected_shape)
            else:
                mfcc = np.zeros(expected_shape, dtype=np.float32)

        if self.augment:
            # Mixup
            mixup_prob = 0.5 if class_rarity in ("very_rare", "rare") else \
                         0.2 if class_rarity == "medium" else 0.0
            mixup_alpha = 0.3 if class_rarity in ("very_rare", "rare") else 0.2

            if random.random() < mixup_prob:
                other_idx = self._get_random_same_class(label, idx)
                if other_idx is not None:
                    other_track_id = self.indices[other_idx]
                    other_crops = self.extractor.prepare_crops_for_track(
                        other_track_id, target_frames=self.target_frames,
                        n_crops=1, mode="train",
                    )
                    if other_crops and other_crops[0] is not None:
                        other_mfcc = self._compute_deltas_on_the_fly(other_crops[0])
                        mfcc = MFCCAugmentation.mixup_same_class(mfcc, other_mfcc, alpha=mixup_alpha)

            # Curriculum augmentation
            mfcc = MFCCAugmentation.apply_curriculum_augmentation(
                mfcc, class_rarity,
                epoch=self.current_epoch,
                total_epochs=self.total_epochs,
            )

        mfcc_tensor = torch.from_numpy(mfcc.copy()).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return mfcc_tensor, label_tensor

    def _record_failure(self, track_id: int, status: Dict[str, Any]) -> None:
        if track_id in self.failed_tracks:
            return
        self.failed_tracks.append(track_id)
        self.failed_tracks_details[track_id] = {
            "error_type": status.get("error_type", "unknown"),
            "error_message": status.get("error_message", ""),
        }
        if self.track_failure_handler:
            try:
                self.track_failure_handler(track_id, status)
            except Exception:
                logger.warning("Failure handler raised for track %d", track_id)

    def get_failed_tracks_report(self) -> Dict[str, Any]:
        n_total = len(self.indices)
        n_failed = len(self.failed_tracks)
        success_rate = (n_total - n_failed) / n_total if n_total > 0 else 0.0
        if n_failed > 0:
            logger.warning("Failure report: %d/%d failed (%.2f%% success)",
                           n_failed, n_total, success_rate * 100)
        return {
            "failed_count": n_failed, "failed_tracks": self.failed_tracks,
            "failed_details": self.failed_tracks_details,
            "total_tracks": n_total, "success_rate": success_rate,
        }

    def get_successful_indices(self) -> List[int]:
        return [idx for idx in self.indices if idx not in self.failed_tracks]

    def get_failed_indices(self) -> List[int]:
        return self.failed_tracks.copy()

    def clear_failed_tracks(self) -> None:
        self.failed_tracks.clear()
        self.failed_tracks_details.clear()

    def get_cached_count(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_sample_shape(self) -> Tuple[int, int, int]:
        return (self.n_channels, self.n_mfcc, self.target_frames)


def create_mfcc_dataloaders(
    subset: str = "medium",
    min_samples_per_genre: int = 10,
    dataset_id: Optional[str] = None,
    batch_size: int = 32,
    target_frames: int = 430,
    num_workers: int = 0,
    cache_mfcc: bool = True,
    use_disk_cache: bool = True,
    augment_train: bool = True,
    use_balanced_sampling: bool = True,
    very_rare_threshold: int = 100,
    rare_threshold: int = 500,
    medium_threshold: int = 1400,
    failure_handler: Optional[Callable] = None,
    n_mfcc: int = 40,
    use_deltas: bool = True,
    use_spectral_features: bool = False,
    use_chroma: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, Any], Dict[int, int]]:
    """Create train/val/test DataLoaders."""
    logger.info("=" * 60)
    logger.info("CREATING MFCC DATALOADERS")
    logger.info("=" * 60)

    data = load_data(subset=subset, min_samples_per_genre=min_samples_per_genre,
                     dataset_id=dataset_id)
    train_indices, val_indices, test_indices = load_track_indices(
        subset=subset, min_samples_per_genre=min_samples_per_genre, dataset_id=dataset_id)
    genre_names: List[str] = data["genre_names"]

    train_idx_list = train_indices.tolist() if hasattr(train_indices, "tolist") else list(train_indices)
    val_idx_list = val_indices.tolist() if hasattr(val_indices, "tolist") else list(val_indices)
    test_idx_list = test_indices.tolist() if hasattr(test_indices, "tolist") else list(test_indices)

    y_train_list = data["y_train"].tolist() if hasattr(data["y_train"], "tolist") else list(data["y_train"])
    y_val_list = data["y_val"].tolist() if hasattr(data["y_val"], "tolist") else list(data["y_val"])
    y_test_list = data["y_test"].tolist() if hasattr(data["y_test"], "tolist") else list(data["y_test"])

    assert len(train_idx_list) == len(y_train_list)
    assert len(val_idx_list) == len(y_val_list)
    assert len(test_idx_list) == len(y_test_list)

    y_train_array = np.array(y_train_list)
    class_counts = {}
    for class_idx in range(len(genre_names)):
        count = int((y_train_array == class_idx).sum())
        if count > 0:
            class_counts[class_idx] = count

    vr = sum(1 for c in class_counts.values() if c <= very_rare_threshold)
    r = sum(1 for c in class_counts.values() if very_rare_threshold < c <= rare_threshold)
    m = sum(1 for c in class_counts.values() if rare_threshold < c <= medium_threshold)
    c = sum(1 for c in class_counts.values() if c > medium_threshold)

    logger.info("Dataset splits — train: %d, val: %d, test: %d, genres: %d",
                len(train_idx_list), len(val_idx_list), len(test_idx_list), len(genre_names))
    logger.info("Class distribution — very_rare(≤%d):%d, rare(≤%d):%d, medium(≤%d):%d, common:%d",
                very_rare_threshold, vr, rare_threshold, r, medium_threshold, m, c)

    mfcc_config = MFCCConfig(
        include_delta=use_deltas, include_delta2=use_deltas,
        n_mfcc=n_mfcc, use_spectral_features=use_spectral_features, use_chroma=use_chroma)
    n_channels = mfcc_config.get_total_channels()

    logger.info("MFCC Configuration — n_mfcc=%d, deltas=%s → %d channels",
                n_mfcc, use_deltas, n_channels)

    shared_loader = AudioLoader()

    train_dataset = MFCCDataset(
        indices=np.array(train_idx_list), labels=np.array(y_train_list),
        audio_loader=shared_loader, mfcc_config=mfcc_config,
        target_frames=target_frames, cache_mfcc=cache_mfcc, use_disk_cache=use_disk_cache,
        augment=augment_train, class_counts=class_counts,
        very_rare_threshold=very_rare_threshold,
        rare_threshold=rare_threshold,
        medium_threshold=medium_threshold,
        track_failure_handler=failure_handler,
    )

    val_dataset = MFCCDataset(
        indices=np.array(val_idx_list), labels=np.array(y_val_list),
        audio_loader=shared_loader, mfcc_config=mfcc_config,
        target_frames=target_frames, cache_mfcc=cache_mfcc, use_disk_cache=use_disk_cache,
        augment=False, class_counts=class_counts,
        very_rare_threshold=very_rare_threshold,
        rare_threshold=rare_threshold,
        medium_threshold=medium_threshold,
    )

    test_dataset = MFCCDataset(
        indices=np.array(test_idx_list), labels=np.array(y_test_list),
        audio_loader=shared_loader, mfcc_config=mfcc_config,
        target_frames=target_frames, cache_mfcc=cache_mfcc, use_disk_cache=use_disk_cache,
        augment=False, class_counts=class_counts,
        very_rare_threshold=very_rare_threshold,
        rare_threshold=rare_threshold,
        medium_threshold=medium_threshold,
    )

    sample_shape = train_dataset.get_sample_shape()
    logger.info("Dataset sample shape: (channels=%d, mfcc=%d, frames=%d)",
                sample_shape[0], sample_shape[1], sample_shape[2])

    if use_balanced_sampling and augment_train:
        sample_weights = np.array([1.0 / class_counts.get(label, 1) for label in y_train_list])
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(y_train_list), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers, pin_memory=True)
        logger.info("Using WeightedRandomSampler for balanced training batches")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info("DataLoaders created — train: %d batches, val: %d, test: %d",
                len(train_loader), len(val_loader), len(test_loader))

    failure_report = {
        "train": train_dataset.get_failed_tracks_report(),
        "val": val_dataset.get_failed_tracks_report(),
        "test": test_dataset.get_failed_tracks_report(),
    }

    total_failed = (failure_report["train"]["failed_count"] +
                    failure_report["val"]["failed_count"] +
                    failure_report["test"]["failed_count"])
    if total_failed > 0:
        logger.warning("Total failed tracks: %d", total_failed)
    else:
        logger.info("All tracks loaded successfully")

    try:
        sample_batch, _ = next(iter(train_loader))
        logger.info("Example batch — MFCC shape: %s", sample_batch.shape)
    except StopIteration:
        logger.warning("Training loader is empty")

    logger.info("=" * 60)
    return train_loader, val_loader, test_loader, genre_names, failure_report, class_counts


if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    setup_logging(level=logging.INFO, mode="console")

    train_loader, val_loader, test_loader, genres, report, class_counts = \
        create_mfcc_dataloaders(
            subset="medium", min_samples_per_genre=10,
            batch_size=4, target_frames=430,
            augment_train=True, use_balanced_sampling=True,
        )

    print(f"\nGenres: {genres[:5]}...")
    for split, r in report.items():
        print(f"  {split}: {r['failed_count']}/{r['total_tracks']} ({r['success_rate']:.2%})")

    for mfcc, labels in train_loader:
        print(f"\nSample batch — MFCC shape: {mfcc.shape}, labels: {labels}")
        break