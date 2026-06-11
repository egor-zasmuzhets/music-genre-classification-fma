"""
MFCC feature extraction from audio with multi-level disk caching.

Extracts MFCC features from raw audio waveforms, optionally including
delta and delta-delta coefficients. Supports disk caching of base MFCC
slices with deterministic crop positions for reproducibility.
Tracks known-bad audio files to avoid redundant extraction attempts.

Typical usage:
    from src.data.mfcc_extractor import MFCCExtractor, MFCCConfig

    config = MFCCConfig(n_mfcc=40, include_delta=True, include_delta2=True)
    extractor = MFCCExtractor(config=config)

    # Получить 3 среза для train
    crops = extractor.prepare_crops_for_track(2, target_frames=430, n_crops=3, mode="train")

    # Получить 1 центральный срез для eval
    crop = extractor.prepare_crops_for_track(2, target_frames=430, n_crops=1, mode="eval")
"""

import hashlib
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from src.utils.config import paths, audio_params
from src.data.audio_loader import AudioLoader


logger = logging.getLogger(__name__)


class MFCCConfig:
    """
    Configuration for MFCC feature extraction.

    Parameters from configs/audio.yaml are used as defaults for any
    values not explicitly provided. Delta and delta-delta flags are
    MFCC-specific and do not come from the audio config.

    Attributes:
        include_delta: Whether to compute first-order delta features.
        include_delta2: Whether to compute second-order delta features.
        n_mfcc: Number of MFCC coefficients to extract.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        win_length: Window length for STFT.
        sr: Target sample rate.
        duration: Target audio duration in seconds.
        use_spectral_features: Whether to add spectral features.
        use_chroma: Whether to add chroma features.
    """

    def __init__(
        self,
        include_delta: bool = True,
        include_delta2: bool = True,
        n_mfcc: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        sr: Optional[int] = None,
        duration: Optional[float] = None,
        use_spectral_features: bool = False,
        use_chroma: bool = False,
    ) -> None:
        """
        Initialize MFCC configuration.

        Args:
            include_delta: Include first-order temporal deltas.
            include_delta2: Include second-order temporal deltas.
            n_mfcc: Number of MFCC coefficients. Defaults to 40.
            n_fft: FFT window size. Defaults to audio_params.n_fft.
            hop_length: Hop length in samples. Defaults to audio_params.hop_length.
            win_length: Window length in samples. Defaults to audio_params.win_length.
            sr: Sample rate. Defaults to audio_params.sample_rate.
            duration: Audio duration in seconds. Defaults to audio_params.duration.
            use_spectral_features: Add spectral features (centroid, bandwidth,
                                   rolloff, ZCR, RMS).
            use_chroma: Add chroma features (pitch class profile).
        """
        self.include_delta = include_delta
        self.include_delta2 = include_delta2
        self.n_mfcc = n_mfcc if n_mfcc is not None else 40
        self.n_fft = n_fft if n_fft is not None else audio_params.n_fft
        self.hop_length = hop_length if hop_length is not None else audio_params.hop_length
        self.win_length = win_length if win_length is not None else audio_params.win_length
        self.sr = sr if sr is not None else audio_params.sample_rate
        self.duration = duration if duration is not None else audio_params.duration
        self.use_spectral_features = use_spectral_features
        self.use_chroma = use_chroma

    def get_cache_key(self) -> str:
        """Generate a unique cache key based on configuration parameters."""
        data = {
            "n_mfcc": self.n_mfcc,
            "include_delta": self.include_delta,
            "include_delta2": self.include_delta2,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_fft": self.n_fft,
            "use_spectral_features": self.use_spectral_features,
            "use_chroma": self.use_chroma,
        }
        key_str = str(sorted(data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Export librosa-compatible parameter dictionary."""
        return {
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
        }

    def get_total_channels(self) -> int:
        """Calculate total number of feature channels."""
        n_channels = 1
        if self.include_delta:
            n_channels += 1
        if self.include_delta2:
            n_channels += 1
        if self.use_spectral_features:
            n_channels += 1
        if self.use_chroma:
            n_channels += 1
        return n_channels


class MFCCExtractor:
    """
    Extracts MFCC features from audio tracks with slice-based caching.

    Uses librosa for feature extraction. Caches only base MFCC slices
    (without deltas) to save disk space. Delta and delta-delta features
    are computed on-the-fly during loading.

    Crop positions are deterministic based on track_id for reproducibility.

    Attributes:
        config: MFCCConfig with extraction parameters.
        audio_loader: AudioLoader instance for waveform access.
        use_disk_cache: Whether disk caching is enabled.
        cache_subdir: Path to this configuration's cache directory.
    """

    def __init__(
        self,
        config: Optional[MFCCConfig] = None,
        audio_loader: Optional[AudioLoader] = None,
        use_disk_cache: bool = True,
        disk_cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the MFCC extractor.

        Args:
            config: MFCC extraction configuration. Defaults to MFCCConfig().
            audio_loader: AudioLoader for reading waveforms. Creates one if None.
            use_disk_cache: Enable disk caching of extracted features.
            disk_cache_dir: Root directory for MFCC caches.
                            Defaults to paths.mfcc_cache_dir.
        """
        self.config = config or MFCCConfig()
        self.audio_loader = audio_loader or AudioLoader()
        self.use_disk_cache = use_disk_cache
        self.disk_cache_dir = disk_cache_dir or paths.mfcc_cache_dir

        if self.use_disk_cache:
            self.cache_subdir = self.disk_cache_dir / self.config.get_cache_key()
            self.cache_subdir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_subdir = None

        self._failed_tracks: set = set()

        n_channels = self.config.get_total_channels()
        total_features = n_channels * self.config.n_mfcc

        logger.info(
            "MFCCExtractor initialized — n_mfcc=%d, deltas: %s/%s, "
            "spectral=%s, chroma=%s → %d channel(s), %d total features, disk_cache=%s",
            self.config.n_mfcc,
            "on" if self.config.include_delta else "off",
            "on" if self.config.include_delta2 else "off",
            "on" if self.config.use_spectral_features else "off",
            "on" if self.config.use_chroma else "off",
            n_channels,
            total_features,
            "on" if use_disk_cache else "off",
        )

    @property
    def failed_track_count(self) -> int:
        """Number of unique tracks that failed to load."""
        return len(self._failed_tracks)

    def clear_failed_tracks(self) -> None:
        """Reset the failed track cache."""
        count = len(self._failed_tracks)
        self._failed_tracks.clear()
        logger.debug("Cleared %d failed track records", count)

    def _get_track_dir(self, track_id: int) -> Path:
        """Get the cache directory for a specific track."""
        return self.cache_subdir / f"{track_id:06d}"

    def _get_slice_cache_path(
        self,
        track_id: int,
        target_frames: int,
        crop_index: int,
        mode: str = "train",
    ) -> Path:
        """
        Build cache path for a specific MFCC slice.

        Args:
            track_id: Numeric track identifier.
            target_frames: Number of time frames in the slice.
            crop_index: Index of the crop (0 for center, 0..N for random).
            mode: 'train' or 'eval' (affects filename).

        Returns:
            Path to the .npy cache file.
        """
        track_dir = self._get_track_dir(track_id)
        if mode == "eval" or crop_index < 0:
            suffix = f"base_{target_frames}_center.npy"
        else:
            suffix = f"base_{target_frames}_crop{crop_index}.npy"
        return track_dir / suffix

    def _compute_crop_start(
        self,
        track_id: int,
        n_frames: int,
        target_frames: int,
        crop_index: int,
        n_crops: int,
        mode: str = "train",
    ) -> int:
        """
        Compute deterministic crop start position for a track.

        Uses track_id as seed for reproducibility.
        Same track_id always produces same crop positions.

        Args:
            track_id: Numeric track identifier.
            n_frames: Total number of MFCC frames in the track.
            target_frames: Desired number of frames per slice.
            crop_index: Which crop (0, 1, 2, ...).
            n_crops: Total number of crops requested.
            mode: 'train' or 'eval'. Eval always returns center.

        Returns:
            Start frame index for the crop.
        """
        if n_frames <= target_frames:
            return 0

        max_start = n_frames - target_frames

        if mode == "eval" or n_crops == 1:
            return max_start // 2

        rng = np.random.RandomState(track_id * 31 + crop_index * 7)

        if n_crops <= 1:
            return max_start // 2

        base_step = max_start / n_crops
        offset_range = int(base_step * 0.25)
        offset = rng.randint(-offset_range, offset_range + 1) if offset_range > 0 else 0
        start = int(crop_index * base_step + offset)

        return max(0, min(max_start, start))

    def _extract_base_mfcc(self, track_id: int) -> Optional[np.ndarray]:
        """
        Extract base MFCC matrix for a track (no deltas, no padding).

        Returns:
            Array of shape (n_mfcc, n_frames) or None on failure.
        """
        if track_id in self._failed_tracks:
            return None

        audio, audio_status = self.audio_loader.load_audio_with_status(
            track_id, self.config.sr, self.config.duration
        )

        if not audio_status["success"]:
            self._failed_tracks.add(track_id)
            logger.warning(
                "Track %d: audio load failed — %s",
                track_id,
                audio_status.get("error_message", "unknown"),
            )
            return None

        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.config.sr,
                **self.config.to_dict(),
            )
            return mfcc.astype(np.float32)
        except Exception as e:
            self._failed_tracks.add(track_id)
            logger.warning("Track %d: MFCC extraction failed — %s", track_id, e)
            return None

    def prepare_crops_for_track(
        self,
        track_id: int,
        target_frames: int = 430,
        n_crops: int = 1,
        mode: str = "eval",
    ) -> List[Optional[np.ndarray]]:
        """
        Prepare MFCC slices for a track with deterministic crop positions.

        Caches only base MFCC slices (n_mfcc, target_frames).
        Deltas and multi-channel stacking are done on-the-fly by the caller.

        Args:
            track_id: Numeric track identifier.
            target_frames: Number of time frames per slice.
            n_crops: Number of slices to extract.
            mode: 'train' for training (random crop positions),
                  'eval' for evaluation (center crop only).

        Returns:
            List of numpy arrays, each shape (n_mfcc, target_frames).
            Failed crops are None.
        """
        if track_id in self._failed_tracks:
            return [None] * n_crops

        results = []

        for crop_idx in range(n_crops):
            cache_path = self._get_slice_cache_path(
                track_id, target_frames, crop_idx if mode == "train" else -1, mode
            )

            if self.use_disk_cache and cache_path.exists():
                try:
                    mfcc_slice = np.load(cache_path)
                    results.append(mfcc_slice)
                    continue
                except Exception:
                    logger.warning(
                        "Corrupt cache for track %d crop %d, re-extracting",
                        track_id, crop_idx,
                    )

            mfcc_full = self._extract_base_mfcc(track_id)
            if mfcc_full is None:
                results.append(None)
                continue

            n_frames = mfcc_full.shape[1]
            start = self._compute_crop_start(
                track_id, n_frames, target_frames, crop_idx, n_crops, mode
            )

            if n_frames < target_frames:
                pad_width = ((0, 0), (0, target_frames - n_frames))
                mfcc_slice = np.pad(mfcc_full, pad_width, mode="constant")
            else:
                mfcc_slice = mfcc_full[:, start:start + target_frames]

            mfcc_slice = mfcc_slice.astype(np.float32)

            if self.use_disk_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    np.save(cache_path, mfcc_slice)
                except Exception:
                    logger.warning(
                        "Failed to write cache for track %d crop %d", track_id, crop_idx
                    )

            results.append(mfcc_slice)

        return results

    def get_base_mfcc_slice(
        self,
        track_id: int,
        target_frames: int = 430,
        crop_index: int = 0,
        mode: str = "eval",
    ) -> Optional[np.ndarray]:
        """
        Get a single base MFCC slice (convenience wrapper).

        Args:
            track_id: Numeric track identifier.
            target_frames: Number of time frames.
            crop_index: Which crop to retrieve.
            mode: 'train' or 'eval'.

        Returns:
            Array of shape (n_mfcc, target_frames) or None.
        """
        crops = self.prepare_crops_for_track(
            track_id, target_frames, n_crops=max(1, crop_index + 1), mode=mode
        )
        if crops and crop_index < len(crops):
            return crops[crop_index]
        return None

    def extract_from_audio(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Dict[str, np.ndarray]:
        """Extract MFCC features from a raw audio waveform."""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, **self.config.to_dict())
        result = {"mfcc": mfcc}

        if self.config.include_delta:
            result["mfcc_delta"] = librosa.feature.delta(mfcc)
        if self.config.include_delta2:
            result["mfcc_delta2"] = librosa.feature.delta(mfcc, order=2)
        if self.config.use_spectral_features:
            spectral_features = [
                librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.config.hop_length),
                librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.config.hop_length),
                librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.config.hop_length),
                librosa.feature.zero_crossing_rate(audio, hop_length=self.config.hop_length),
                librosa.feature.rms(y=audio, hop_length=self.config.hop_length),
            ]
            result["spectral_features"] = np.vstack(spectral_features)
        if self.config.use_chroma:
            result["chroma"] = librosa.feature.chroma_stft(
                y=audio, sr=sr, hop_length=self.config.hop_length, n_chroma=12
            )

        return result

    # ========================================================================
    # Legacy methods (keep for backward compatibility)
    # ========================================================================

    def _get_cache_paths(self, track_id: int) -> Dict[str, Path]:
        """Build paths for all cache tiers of a given track (legacy)."""
        track_str = f"{track_id:06d}"
        base_path = self.cache_subdir / track_str
        return {
            "full": base_path.with_suffix(".full.pkl"),
            "mfcc_base": base_path.with_suffix(".base.npy"),
            "mfcc_with_deltas": base_path.with_suffix(".deltas.npy"),
            "cnn_ready": base_path.with_suffix(".cnn.npy"),
        }

    def _load_from_cache(self, track_id: int, cache_type: str) -> Optional[Any]:
        """Load data from a specific cache tier (legacy)."""
        if not self.use_disk_cache:
            return None
        cache_paths = self._get_cache_paths(track_id)
        cache_path = cache_paths.get(cache_type)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            if cache_type == "full":
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            else:
                return np.load(cache_path)
        except Exception:
            logger.warning("Corrupt cache (%s) for track %d, will re-extract", cache_type, track_id)
            return None

    def _save_to_cache(self, track_id: int, cache_type: str, data: Any) -> None:
        """Save data to a specific cache tier (legacy)."""
        if not self.use_disk_cache:
            return
        cache_paths = self._get_cache_paths(track_id)
        cache_path = cache_paths.get(cache_type)
        if cache_path is None:
            return
        try:
            if cache_type == "full":
                with open(cache_path, "wb") as f:
                    pickle.dump(data, f)
            else:
                np.save(cache_path, data)
        except Exception:
            logger.warning("Failed to write %s cache for track %d", cache_type, track_id)

    def extract_from_track_id_with_status(
        self, track_id: int, use_cache: bool = True,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
        """Extract full MFCC feature dict for a track (legacy)."""
        status: Dict[str, Any] = {
            "success": False, "error_type": None, "error_message": None,
            "track_id": track_id, "loaded_from_cache": False,
        }
        if track_id in self._failed_tracks:
            status["error_type"] = "cached_failure"
            status["error_message"] = "Track previously failed, skipping"
            return None, status
        if use_cache and self.use_disk_cache:
            cached = self._load_from_cache(track_id, "full")
            if cached is not None:
                status["success"] = True
                status["loaded_from_cache"] = True
                status["cache_type"] = "disk"
                return cached, status
        audio, audio_status = self.audio_loader.load_audio_with_status(
            track_id, self.config.sr, self.config.duration
        )
        if not audio_status["success"]:
            self._failed_tracks.add(track_id)
            status["error_type"] = audio_status["error_type"]
            status["error_message"] = f"Audio load failed: {audio_status['error_message']}"
            logger.warning("Track %d: audio load failed — %s", track_id, status["error_message"])
            return None, status
        try:
            features = self.extract_from_audio(audio, self.config.sr)
            if use_cache and self.use_disk_cache:
                self._save_to_cache(track_id, "full", features)
            status["success"] = True
            return features, status
        except Exception as e:
            self._failed_tracks.add(track_id)
            status["error_type"] = "mfcc_extraction_error"
            status["error_message"] = str(e)
            logger.warning("Track %d: MFCC extraction failed — %s", track_id, e)
            return None, status

    def extract_from_track_id(self, track_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Extract full MFCC feature dict (legacy wrapper)."""
        features, _ = self.extract_from_track_id_with_status(track_id)
        return features

    def prepare_for_cnn_with_status(
        self, track_id: int, target_frames: int = 128, use_cache: bool = True,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Prepare MFCC for CNN (legacy, kept for backward compat)."""
        if track_id in self._failed_tracks:
            status = {
                "success": False, "error_type": "cached_failure",
                "error_message": "Track previously failed, skipping",
                "track_id": track_id, "loaded_from_cache": False,
            }
            return None, status
        if use_cache and self.use_disk_cache:
            cached = self._load_from_cache(track_id, "cnn_ready")
            if cached is not None and cached.shape[-1] == target_frames:
                status = {
                    "success": True, "loaded_from_cache": True,
                    "cache_type": "disk", "track_id": track_id,
                }
                return cached, status
        features, status = self.extract_from_track_id_with_status(track_id, use_cache)
        if not status["success"] or features is None:
            self._failed_tracks.add(track_id)
            status["success"] = False
            status["error_type"] = "no_data"
            status["error_message"] = "No feature data returned"
            return None, status
        n_mfcc = self.config.n_mfcc
        n_frames = features["mfcc"].shape[1]
        components = []
        mfcc_components = [features["mfcc"]]
        if self.config.include_delta and "mfcc_delta" in features:
            mfcc_components.append(features["mfcc_delta"])
        if self.config.include_delta2 and "mfcc_delta2" in features:
            mfcc_components.append(features["mfcc_delta2"])
        mfcc_stacked = np.vstack(mfcc_components)
        n_delta_channels = len(mfcc_components)
        mfcc_reshaped = mfcc_stacked.reshape(n_delta_channels, n_mfcc, n_frames)
        for i in range(n_delta_channels):
            components.append(mfcc_reshaped[i:i+1, :, :])
        if self.config.use_spectral_features and "spectral_features" in features:
            spectral = features["spectral_features"]
            spectral_padded = np.zeros((n_mfcc, n_frames))
            n_spectral = min(spectral.shape[0], n_mfcc)
            spectral_padded[:n_spectral, :] = spectral[:n_spectral, :]
            components.append(spectral_padded[np.newaxis, :, :])
        if self.config.use_chroma and "chroma" in features:
            chroma = features["chroma"]
            chroma_padded = np.zeros((n_mfcc, n_frames))
            n_chroma = min(chroma.shape[0], n_mfcc)
            chroma_padded[:n_chroma, :] = chroma[:n_chroma, :]
            components.append(chroma_padded[np.newaxis, :, :])
        cnn_ready = np.vstack(components).astype(np.float32)
        n_frames = cnn_ready.shape[2]
        if n_frames < target_frames:
            pad_width = ((0, 0), (0, 0), (0, target_frames - n_frames))
            cnn_ready = np.pad(cnn_ready, pad_width, mode="constant")
        elif n_frames > target_frames:
            start = (n_frames - target_frames) // 2
            cnn_ready = cnn_ready[:, :, start:start + target_frames]
        if use_cache and self.use_disk_cache:
            self._save_to_cache(track_id, "cnn_ready", cnn_ready)
        return cnn_ready, status

    def prepare_for_cnn(self, track_id: int, target_frames: int = 128) -> Optional[np.ndarray]:
        """Prepare CNN-ready MFCC (legacy wrapper)."""
        mfcc, _ = self.prepare_for_cnn_with_status(track_id, target_frames)
        return mfcc

    def get_mfcc_matrix_with_status(
        self, track_id: int, flatten: bool = False, use_cache: bool = True,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Get base MFCC matrix (legacy)."""
        if track_id in self._failed_tracks:
            return None, {
                "success": False, "error_type": "cached_failure",
                "error_message": "Track previously failed, skipping",
                "track_id": track_id, "loaded_from_cache": False,
            }
        if use_cache and self.use_disk_cache:
            cached = self._load_from_cache(track_id, "mfcc_base")
            if cached is not None:
                status = {"success": True, "loaded_from_cache": True, "cache_type": "disk", "track_id": track_id}
                return cached.flatten() if flatten else cached, status
        features, status = self.extract_from_track_id_with_status(track_id, use_cache)
        if not status["success"]:
            return None, status
        mfcc = features["mfcc"]
        if use_cache and self.use_disk_cache:
            self._save_to_cache(track_id, "mfcc_base", mfcc)
        return mfcc.flatten() if flatten else mfcc, status

    def get_mfcc_matrix(self, track_id: int, flatten: bool = False, use_cache: bool = True) -> Optional[np.ndarray]:
        """Get base MFCC matrix (legacy wrapper)."""
        mfcc, _ = self.get_mfcc_matrix_with_status(track_id, flatten, use_cache)
        return mfcc

    def get_mfcc_with_deltas_with_status(
        self, track_id: int, flatten: bool = False, use_cache: bool = True,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Get stacked MFCC + deltas (legacy)."""
        if track_id in self._failed_tracks:
            return None, {
                "success": False, "error_type": "cached_failure",
                "error_message": "Track previously failed, skipping",
                "track_id": track_id, "loaded_from_cache": False,
            }
        if use_cache and self.use_disk_cache:
            cached = self._load_from_cache(track_id, "mfcc_with_deltas")
            if cached is not None:
                status = {"success": True, "loaded_from_cache": True, "cache_type": "disk", "track_id": track_id}
                return cached.flatten() if flatten else cached, status
        features, status = self.extract_from_track_id_with_status(track_id, use_cache)
        if not status["success"]:
            return None, status
        matrices = [features["mfcc"]]
        if self.config.include_delta and "mfcc_delta" in features:
            matrices.append(features["mfcc_delta"])
        if self.config.include_delta2 and "mfcc_delta2" in features:
            matrices.append(features["mfcc_delta2"])
        combined = np.vstack(matrices)
        if use_cache and self.use_disk_cache:
            self._save_to_cache(track_id, "mfcc_with_deltas", combined)
        return combined.flatten() if flatten else combined, status

    def get_mfcc_with_deltas(self, track_id: int, flatten: bool = False, use_cache: bool = True) -> Optional[np.ndarray]:
        """Get stacked MFCC + deltas (legacy wrapper)."""
        mfcc, _ = self.get_mfcc_with_deltas_with_status(track_id, flatten, use_cache)
        return mfcc

    def extract_batch_with_status(
        self, track_ids: List[int], target_frames: int = 128,
    ) -> Tuple[Dict[int, Optional[np.ndarray]], Dict[int, Dict[str, Any]]]:
        """Extract CNN-ready MFCC for a batch (legacy)."""
        results: Dict[int, Optional[np.ndarray]] = {}
        status_results: Dict[int, Dict[str, Any]] = {}
        total = len(track_ids)
        cache_hits = 0
        cache_misses = 0
        failed: List[int] = []
        for i, track_id in enumerate(track_ids):
            if total >= 500 and (i + 1) % 100 == 0:
                hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
                logger.info("Batch MFCC progress: %d/%d tracks (failed: %d, cache hit: %.1f%%)",
                            i + 1, total, len(failed), 100 * hit_rate)
            mfcc, status = self.prepare_for_cnn_with_status(track_id, target_frames)
            results[track_id] = mfcc
            status_results[track_id] = status
            if status.get("loaded_from_cache", False):
                cache_hits += 1
            else:
                cache_misses += 1
            if not status["success"]:
                failed.append(track_id)
        n_failed = len(failed)
        n_success = total - n_failed
        hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
        logger.info("Batch MFCC complete: %d/%d tracks extracted (cache hit: %.1f%%, failed: %d)",
                    n_success, total, 100 * hit_rate, n_failed)
        if n_failed > 0:
            logger.warning("Failed tracks in batch: %s", failed[:10] if len(failed) > 10 else failed)
        return results, status_results

    def get_failed_tracks_report(self, status_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate error statistics (legacy)."""
        failed_by_type: Dict[str, List[int]] = {}
        failed_tracks: List[int] = []
        for track_id, status in status_results.items():
            if not status["success"]:
                failed_tracks.append(track_id)
                error_type = status.get("error_type", "unknown")
                if error_type not in failed_by_type:
                    failed_by_type[error_type] = []
                failed_by_type[error_type].append(track_id)
        total = len(status_results)
        n_failed = len(failed_tracks)
        return {
            "total_processed": total, "failed_count": n_failed,
            "failed_tracks": failed_tracks, "failed_by_type": failed_by_type,
            "success_rate": (total - n_failed) / total if total > 0 else 0.0,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Collect statistics about the disk cache."""
        if not self.use_disk_cache or not self.cache_subdir.exists():
            return {"enabled": False}
        cache_files = list(self.cache_subdir.glob("*"))
        cache_size_bytes = sum(f.stat().st_size for f in cache_files)
        return {
            "enabled": True,
            "cache_dir": str(self.cache_subdir),
            "total_files": len(cache_files),
            "total_size_mb": cache_size_bytes / (1024 ** 2),
        }

    def print_cache_stats(self) -> None:
        """Print a human-readable summary of the disk cache state."""
        stats = self.get_cache_stats()
        if not stats["enabled"]:
            print("Disk cache is disabled.")
            return
        print("=" * 60)
        print("MFCC CACHE STATISTICS")
        print("=" * 60)
        print(f"Directory:    {stats['cache_dir']}")
        print(f"Total files:  {stats['total_files']}")
        print(f"Total size:   {stats['total_size_mb']:.2f} MB")

    def clear_cache(self) -> None:
        """Remove all cached MFCC features for this configuration."""
        if not self.use_disk_cache or not self.cache_subdir.exists():
            return
        file_count = len(list(self.cache_subdir.glob("*")))
        size_mb = sum(f.stat().st_size for f in self.cache_subdir.glob("*")) / (1024 ** 2)
        shutil.rmtree(self.cache_subdir)
        self.cache_subdir.mkdir(parents=True, exist_ok=True)
        self._failed_tracks.clear()
        logger.info("MFCC cache cleared: %s (%d files, %.1f MB freed, failed tracks reset)",
                     self.cache_subdir, file_count, size_mb)


if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    setup_logging(level=logging.INFO, mode="console")

    extractor = MFCCExtractor()
    extractor.print_cache_stats()

    test_tracks = [2, 3, 5]
    print(f"\nTest tracks: {test_tracks}")

    for track_id in test_tracks:
        print(f"\n--- Track {track_id} ---")
        crops = extractor.prepare_crops_for_track(track_id, target_frames=430, n_crops=3, mode="train")
        for i, crop in enumerate(crops):
            if crop is not None:
                print(f"  Crop {i}: shape={crop.shape}, mean={crop.mean():.4f}, std={crop.std():.4f}")
            else:
                print(f"  Crop {i}: FAILED")

    print()
    extractor.print_cache_stats()