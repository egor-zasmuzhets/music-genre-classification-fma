"""
src/data/mfcc_extractor.py
MFCC feature extraction from audio with multi-level disk caching.

Extracts MFCC features from raw audio waveforms, optionally including
delta and delta-delta coefficients. Supports four-tier disk caching
(full feature dict, base MFCC, MFCC with deltas, CNN-ready) to
accelerate repeated access during training. Tracks known-bad audio files
to avoid redundant extraction attempts.

Typical usage:
    from src.data.mfcc_extractor import MFCCExtractor, MFCCConfig

    config = MFCCConfig(include_delta=True, include_delta2=True)
    extractor = MFCCExtractor(config=config)

    mfcc, status = extractor.prepare_for_cnn_with_status(2, target_frames=128)

    results, statuses = extractor.extract_batch_with_status(
        [2, 3, 5], target_frames=128
    )
"""

import hashlib
import logging
import pickle
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
    ):
        """
        Initialize MFCC configuration.

        Args:
            include_delta: Include first-order temporal deltas.
            include_delta2: Include second-order temporal deltas.
            n_mfcc: Number of MFCC coefficients. Defaults to audio_params.n_mfcc.
            n_fft: FFT window size. Defaults to audio_params.n_fft.
            hop_length: Hop length in samples. Defaults to audio_params.hop_length.
            win_length: Window length in samples. Defaults to audio_params.win_length.
            sr: Sample rate. Defaults to audio_params.sample_rate.
            duration: Audio duration in seconds. Defaults to audio_params.duration.
        """
        self.include_delta = include_delta
        self.include_delta2 = include_delta2
        self.n_mfcc = n_mfcc if n_mfcc is not None else audio_params.n_mfcc
        self.n_fft = n_fft if n_fft is not None else audio_params.n_fft
        self.hop_length = hop_length if hop_length is not None else audio_params.hop_length
        self.win_length = win_length if win_length is not None else audio_params.win_length
        self.sr = sr if sr is not None else audio_params.sample_rate
        self.duration = duration if duration is not None else audio_params.duration

    def get_cache_key(self) -> str:
        """
        Generate a unique cache key based on configuration parameters.

        Uses MD5 hash of sorted parameter key-value pairs to produce
        a short, deterministic string for cache directory naming.

        Returns:
            16-character hex digest string.
        """
        data = {
            "n_mfcc": self.n_mfcc,
            "include_delta": self.include_delta,
            "include_delta2": self.include_delta2,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_fft": self.n_fft,
        }
        key_str = str(sorted(data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """
        Export librosa-compatible parameter dictionary.

        Returns:
            Dictionary with keys n_mfcc, n_fft, hop_length, win_length.
        """
        return {
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
        }


class MFCCExtractor:
    """
    Extracts MFCC features from audio tracks with multi-tier caching.

    Uses librosa for feature extraction and supports four levels of
    disk caching to trade off storage for speed:
    - 'full': complete feature dictionary as pickle
    - 'mfcc_base': base MFCC matrix as .npy
    - 'mfcc_with_deltas': vertically stacked MFCC + deltas as .npy
    - 'cnn_ready': time-normalized (padded/cropped) matrix as .npy

    Each configuration gets its own cache subdirectory identified
    by a hash of key parameters. Tracks that fail to load are remembered
    to avoid repeated extraction attempts during multi-epoch training.

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
    ):
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
            logger.debug("MFCC cache directory: %s", self.cache_subdir)
        else:
            self.cache_subdir = None

        self._failed_tracks: set = set()

        n_channels = 1
        if self.config.include_delta:
            n_channels += 1
        if self.config.include_delta2:
            n_channels += 1

        logger.info(
            "MFCCExtractor initialized — n_mfcc=%d, deltas: %s/%s "
            "→ %d channel(s), disk_cache=%s",
            self.config.n_mfcc,
            "on" if self.config.include_delta else "off",
            "on" if self.config.include_delta2 else "off",
            n_channels,
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

    def _get_cache_paths(self, track_id: int) -> Dict[str, Path]:
        """
        Build paths for all cache tiers of a given track.

        Args:
            track_id: Numeric track identifier.

        Returns:
            Dictionary mapping cache type names to file paths:
            - 'full': pickle with complete feature dict
            - 'mfcc_base': .npy with base MFCC matrix
            - 'mfcc_with_deltas': .npy with vertically stacked MFCC+deltas
            - 'cnn_ready': .npy with time-normalized matrix
        """
        track_str = f"{track_id:06d}"
        base_path = self.cache_subdir / track_str

        return {
            "full": base_path.with_suffix(".full.pkl"),
            "mfcc_base": base_path.with_suffix(".base.npy"),
            "mfcc_with_deltas": base_path.with_suffix(".deltas.npy"),
            "cnn_ready": base_path.with_suffix(".cnn.npy"),
        }

    def _load_from_cache(self, track_id: int, cache_type: str) -> Optional[Any]:
        """
        Load data from a specific cache tier.

        Args:
            track_id: Numeric track identifier.
            cache_type: One of 'full', 'mfcc_base', 'mfcc_with_deltas', 'cnn_ready'.

        Returns:
            Cached data (dict or np.ndarray), or None if not found or corrupt.
        """
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
            logger.debug(
                "Corrupt cache (%s) for track %d, will re-extract",
                cache_type,
                track_id,
            )
            return None

    def _save_to_cache(self, track_id: int, cache_type: str, data: Any) -> None:
        """
        Save data to a specific cache tier.

        Args:
            track_id: Numeric track identifier.
            cache_type: Cache tier name.
            data: Data to persist (dict for 'full', np.ndarray otherwise).
        """
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
            logger.warning(
                "Failed to write %s cache for track %d", cache_type, track_id
            )

    def extract_from_audio(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Dict[str, np.ndarray]:
        """
        Extract MFCC features from a raw audio waveform.

        Args:
            audio: 1D numpy array of float32 audio samples.
            sr: Sample rate of the audio.

        Returns:
            Dictionary with keys:
            - 'mfcc': (n_mfcc, n_frames) base MFCC array
            - 'mfcc_delta': (n_mfcc, n_frames) first-order deltas (if enabled)
            - 'mfcc_delta2': (n_mfcc, n_frames) second-order deltas (if enabled)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            **self.config.to_dict(),
        )

        result = {"mfcc": mfcc}

        if self.config.include_delta:
            result["mfcc_delta"] = librosa.feature.delta(mfcc)

        if self.config.include_delta2:
            result["mfcc_delta2"] = librosa.feature.delta(mfcc, order=2)

        return result

    def extract_from_track_id_with_status(
        self,
        track_id: int,
        use_cache: bool = True,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
        """
        Extract full MFCC feature dict for a track with status reporting.

        Args:
            track_id: Numeric track identifier.
            use_cache: Whether to check and populate disk cache.

        Returns:
            Tuple of (features_dict_or_None, status_dict).
            status_dict keys: success, error_type, error_message,
            loaded_from_cache, track_id.
        """
        status: Dict[str, Any] = {
            "success": False,
            "error_type": None,
            "error_message": None,
            "track_id": track_id,
            "loaded_from_cache": False,
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
            status["error_message"] = (
                f"Audio load failed: {audio_status['error_message']}"
            )
            logger.warning(
                "Track %d: audio load failed — %s",
                track_id,
                status["error_message"],
            )
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
            logger.warning(
                "Track %d: MFCC extraction failed — %s",
                track_id,
                e,
            )
            return None, status

    def extract_from_track_id(
        self,
        track_id: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract full MFCC feature dict (compatibility wrapper).

        Prefer extract_from_track_id_with_status for new code.

        Args:
            track_id: Numeric track identifier.

        Returns:
            Feature dictionary or None on failure.
        """
        features, _ = self.extract_from_track_id_with_status(track_id)
        return features

    def get_mfcc_matrix_with_status(
        self,
        track_id: int,
        flatten: bool = False,
        use_cache: bool = True,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Get base MFCC matrix with status reporting.

        Args:
            track_id: Numeric track identifier.
            flatten: If True, flatten to 1D array.
            use_cache: Whether to use disk cache.

        Returns:
            Tuple of (mfcc_matrix_or_None, status_dict).
        """
        if track_id in self._failed_tracks:
            status = {
                "success": False,
                "error_type": "cached_failure",
                "error_message": "Track previously failed, skipping",
                "track_id": track_id,
                "loaded_from_cache": False,
            }
            return None, status

        if use_cache and self.use_disk_cache:
            cached = self._load_from_cache(track_id, "mfcc_base")
            if cached is not None:
                status = {
                    "success": True,
                    "loaded_from_cache": True,
                    "cache_type": "disk",
                    "track_id": track_id,
                }
                result = cached.flatten() if flatten else cached
                return result, status

        features, status = self.extract_from_track_id_with_status(track_id, use_cache)

        if not status["success"]:
            return None, status

        mfcc = features["mfcc"]

        if use_cache and self.use_disk_cache:
            self._save_to_cache(track_id, "mfcc_base", mfcc)

        if flatten:
            return mfcc.flatten(), status
        return mfcc, status

    def get_mfcc_matrix(
        self,
        track_id: int,
        flatten: bool = False,
        use_cache: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Get base MFCC matrix (compatibility wrapper).

        Prefer get_mfcc_matrix_with_status for new code.
        """
        mfcc, _ = self.get_mfcc_matrix_with_status(track_id, flatten, use_cache)
        return mfcc

    def get_mfcc_with_deltas_with_status(
        self,
        track_id: int,
        flatten: bool = False,
        use_cache: bool = True,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Get vertically stacked MFCC + deltas matrix with status.

        Args:
            track_id: Numeric track identifier.
            flatten: If True, flatten to 1D array.
            use_cache: Whether to use disk cache.

        Returns:
            Tuple of (combined_matrix_or_None, status_dict).
        """
        if track_id in self._failed_tracks:
            status = {
                "success": False,
                "error_type": "cached_failure",
                "error_message": "Track previously failed, skipping",
                "track_id": track_id,
                "loaded_from_cache": False,
            }
            return None, status

        if use_cache and self.use_disk_cache:
            cached = self._load_from_cache(track_id, "mfcc_with_deltas")
            if cached is not None:
                status = {
                    "success": True,
                    "loaded_from_cache": True,
                    "cache_type": "disk",
                    "track_id": track_id,
                }
                result = cached.flatten() if flatten else cached
                return result, status

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

        if flatten:
            return combined.flatten(), status
        return combined, status

    def get_mfcc_with_deltas(
        self,
        track_id: int,
        flatten: bool = False,
        use_cache: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Get stacked MFCC + deltas (compatibility wrapper).

        Prefer get_mfcc_with_deltas_with_status for new code.
        """
        mfcc, _ = self.get_mfcc_with_deltas_with_status(track_id, flatten, use_cache)
        return mfcc

    def prepare_for_cnn_with_status(
        self,
        track_id: int,
        target_frames: int = 128,
        use_cache: bool = True,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Prepare time-normalized MFCC matrix for CNN input with status.

        Pads or center-crops the time axis to exactly target_frames.
        Result shape is (n_channels, target_frames).

        Args:
            track_id: Numeric track identifier.
            target_frames: Exact number of time frames required.
            use_cache: Whether to use and populate disk cache.

        Returns:
            Tuple of (cnn_ready_matrix_or_None, status_dict).
        """
        if track_id in self._failed_tracks:
            status = {
                "success": False,
                "error_type": "cached_failure",
                "error_message": "Track previously failed, skipping",
                "track_id": track_id,
                "loaded_from_cache": False,
            }
            return None, status

        if use_cache and self.use_disk_cache:
            cached = self._load_from_cache(track_id, "cnn_ready")
            if cached is not None and cached.shape[-1] == target_frames:
                status = {
                    "success": True,
                    "loaded_from_cache": True,
                    "cache_type": "disk",
                    "track_id": track_id,
                }
                return cached, status
            elif cached is not None:
                logger.debug(
                    "Track %d: CNN cache has %d frames, need %d — re-processing",
                    track_id,
                    cached.shape[-1],
                    target_frames,
                )

        combined, status = self.get_mfcc_with_deltas_with_status(
            track_id, flatten=False, use_cache=use_cache
        )

        if not status["success"] or combined is None:
            self._failed_tracks.add(track_id)
            if status.get("success") is None or status["success"] is False:
                pass
            else:
                status["success"] = False
                status["error_type"] = "no_data"
                status["error_message"] = "No MFCC data returned"
            return None, status

        combined = combined.astype(np.float32)
        n_channels, n_frames = combined.shape

        if n_frames < target_frames:
            pad_width = ((0, 0), (0, target_frames - n_frames))
            combined = np.pad(combined, pad_width, mode="constant")
        else:
            start = (n_frames - target_frames) // 2
            combined = combined[:, start : start + target_frames]

        if use_cache and self.use_disk_cache:
            self._save_to_cache(track_id, "cnn_ready", combined)

        return combined, status

    def prepare_for_cnn(
        self,
        track_id: int,
        target_frames: int = 128,
    ) -> Optional[np.ndarray]:
        """
        Prepare CNN-ready MFCC (compatibility wrapper).

        Prefer prepare_for_cnn_with_status for new code.
        """
        mfcc, _ = self.prepare_for_cnn_with_status(track_id, target_frames)
        return mfcc

    def extract_batch_with_status(
        self,
        track_ids: List[int],
        target_frames: int = 128,
    ) -> Tuple[Dict[int, Optional[np.ndarray]], Dict[int, Dict[str, Any]]]:
        """
        Extract CNN-ready MFCC for a batch of tracks.

        Processes tracks sequentially. Logs aggregate cache hit/miss
        statistics and failure summary on completion.

        Args:
            track_ids: List of track identifiers to process.
            target_frames: Target number of time frames per track.

        Returns:
            Tuple of (results_dict, status_dict).
            - results_dict: {track_id: np.ndarray or None}
            - status_dict: {track_id: status_dict}
        """
        results: Dict[int, Optional[np.ndarray]] = {}
        status_results: Dict[int, Dict[str, Any]] = {}
        total = len(track_ids)

        cache_hits = 0
        cache_misses = 0
        failed: List[int] = []

        for i, track_id in enumerate(track_ids):
            if total >= 200 and (i + 1) % 50 == 0:
                hit_rate = (
                    cache_hits / (cache_hits + cache_misses)
                    if (cache_hits + cache_misses) > 0
                    else 0.0
                )
                logger.debug(
                    "Batch MFCC progress: %d/%d tracks "
                    "(failed: %d, cache hit: %.1f%%)",
                    i + 1,
                    total,
                    len(failed),
                    100 * hit_rate,
                )

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
        hit_rate = (
            cache_hits / (cache_hits + cache_misses)
            if (cache_hits + cache_misses) > 0
            else 0.0
        )

        logger.info(
            "Batch MFCC complete: %d/%d tracks extracted "
            "(cache hit: %.1f%%, failed: %d)",
            n_success,
            total,
            100 * hit_rate,
            n_failed,
        )

        if n_failed > 0:
            logger.warning(
                "Failed tracks in batch: %s",
                failed[:10] if len(failed) > 10 else failed,
            )

        return results, status_results

    def get_failed_tracks_report(
        self, status_results: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate error statistics from a batch status dictionary.

        Args:
            status_results: Per-track status dicts from extract_batch_with_status.

        Returns:
            Report with keys: total_processed, failed_count, failed_tracks,
            failed_by_type, success_rate.
        """
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
            "total_processed": total,
            "failed_count": n_failed,
            "failed_tracks": failed_tracks,
            "failed_by_type": failed_by_type,
            "success_rate": (total - n_failed) / total if total > 0 else 0.0,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Collect statistics about the disk cache for this configuration.

        Returns:
            Dictionary with total file count, size in MB, and per-tier
            breakdown. If disk cache is disabled, returns {'enabled': False}.
        """
        if not self.use_disk_cache or not self.cache_subdir.exists():
            return {"enabled": False}

        cache_files = list(self.cache_subdir.glob("*"))
        cache_size_bytes = sum(f.stat().st_size for f in cache_files)

        tiers = {}
        for cache_type in ["full", "base", "deltas", "cnn"]:
            pattern = f"*.{cache_type}.*"
            files = list(self.cache_subdir.glob(pattern))
            tiers[cache_type] = {
                "count": len(files),
                "size_mb": sum(f.stat().st_size for f in files) / (1024 ** 2),
            }

        return {
            "enabled": True,
            "cache_dir": str(self.cache_subdir),
            "total_files": len(cache_files),
            "total_size_mb": cache_size_bytes / (1024 ** 2),
            "by_type": tiers,
        }

    def print_cache_stats(self) -> None:
        """
        Print a human-readable summary of the disk cache state.

        This is a manual debugging/exploration utility.
        """
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
        print("\nBy tier:")
        for cache_type, data in stats["by_type"].items():
            if data["count"] > 0:
                print(f"  {cache_type}: {data['count']} files, "
                      f"{data['size_mb']:.2f} MB")

    def clear_cache(self) -> None:
        """
        Remove all cached MFCC features for this configuration.

        Deletes the entire cache subdirectory and recreates it empty.
        Also clears the failed tracks set.
        """
        if not self.use_disk_cache or not self.cache_subdir.exists():
            logger.debug("No cache to clear (disabled or empty)")
            return

        import shutil

        file_count = len(list(self.cache_subdir.glob("*")))
        size_mb = (
            sum(f.stat().st_size for f in self.cache_subdir.glob("*"))
            / (1024 ** 2)
        )

        shutil.rmtree(self.cache_subdir)
        self.cache_subdir.mkdir(parents=True, exist_ok=True)
        self._failed_tracks.clear()

        logger.info(
            "MFCC cache cleared: %s (%d files, %.1f MB freed, "
            "failed tracks reset)",
            self.cache_subdir,
            file_count,
            size_mb,
        )


if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    setup_logging(level=logging.DEBUG, mode="console")

    extractor = MFCCExtractor()

    extractor.print_cache_stats()

    test_tracks = [2, 3, 5, 10, 20]
    print(f"\nTest tracks: {test_tracks}")

    for track_id in test_tracks:
        print(f"\n--- Track {track_id} ---")
        mfcc = extractor.prepare_for_cnn(track_id, target_frames=128)
        if mfcc is not None:
            print(f"  OK: shape = {mfcc.shape}")
        else:
            print(f"  FAILED")

    print()
    extractor.print_cache_stats()

    print(f"\nFailed tracks recorded: {extractor.failed_track_count}")

    print("\n" + "=" * 60)
    print("Cache file listing for track 2:")
    print("=" * 60)
    cache_paths = extractor._get_cache_paths(2)
    for cache_type, path in cache_paths.items():
        if path.exists():
            print(f"  {cache_type}: {path.name} "
                  f"({path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  {cache_type}: {path.name} (missing)")