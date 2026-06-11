"""
Audio file loader — reads MP3s directly from FMA ZIP archives.

Provides lazy ZIP access, multi-level caching (memory + disk), automatic
resampling to target sample rate and duration, and detailed status reporting
for each track load attempt.

Typical usage:
    from src.data.audio_loader import AudioLoader

    loader = AudioLoader()

    audio, status = loader.load_audio_with_status(track_id=2)
    if status['success']:
        print(f"Loaded: {audio.shape}")

    # Batch loading
    audio_dict, status_dict = loader.load_audio_batch_with_status([2, 3, 5])

    loader.close()
"""

import hashlib
import io
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from src.utils.config import paths, audio_params


logger = logging.getLogger(__name__)


class AudioLoader:
    """
    Loads audio from FMA ZIP archives without full extraction.

    Reads MP3 files directly from ZIP, resamples to the target sample rate,
    trims/pads to the target duration, and caches results in memory and/or
    on disk for faster subsequent access.

    Each load operation returns a status dictionary with detailed error
    information, enabling graceful handling of missing or corrupt tracks.

    Attributes:
        zip_path: Path to the FMA ZIP archive.
        use_cache: Whether in-memory caching is enabled.
        cache_size: Maximum number of tracks to hold in memory.
        use_disk_cache: Whether disk caching is enabled.
        disk_cache_dir: Directory for disk-cached waveforms.
    """

    def __init__(
        self,
        zip_path: Optional[Path] = None,
        cache_size: int = 100,
        use_cache: bool = True,
        use_disk_cache: bool = False,
        disk_cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the audio loader with the specified ZIP archive.

        Args:
            zip_path: Path to the FMA ZIP archive. Defaults to the
                      configured active_zip from configs/paths.yaml.
            cache_size: Maximum number of tracks to cache in memory (LRU-like
                        eviction is not implemented; once full, new tracks
                        are not cached).
            use_cache: Enable in-memory caching of loaded waveforms.
            use_disk_cache: Enable disk caching of loaded waveforms as .npy
                            files for persistence across sessions.
            disk_cache_dir: Directory for disk cache files. Defaults to
                            the configured waveform_cache_dir.

        Raises:
            FileNotFoundError: If the specified ZIP archive does not exist.
        """
        self.zip_path = zip_path or paths.active_zip

        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP archive not found: {self.zip_path}")

        self.use_cache = use_cache
        self.cache_size = cache_size
        self.use_disk_cache = use_disk_cache
        self.disk_cache_dir = disk_cache_dir or paths.waveform_cache_dir

        self._zip: Optional[zipfile.ZipFile] = None
        self._cached_audio: Dict[Tuple, np.ndarray] = {}

        if self.use_disk_cache:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        zip_size_gb = self.zip_path.stat().st_size / (1024 ** 3)

        logger.info(
            "AudioLoader initialized — zip: %s (%.1f GB), "
            "memory_cache: %s (max %d tracks), disk_cache: %s (%s)",
            self.zip_path.name,
            zip_size_gb,
            "on" if use_cache else "off",
            cache_size,
            "on" if use_disk_cache else "off",
            self.disk_cache_dir if use_disk_cache else "n/a",
        )

    @property
    def zip(self) -> zipfile.ZipFile:
        """
        Lazily-opened ZIP archive handle.

        The archive is opened on first access and reused for subsequent calls.
        """
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, 'r')
            logger.debug("ZIP archive opened: %s", self.zip_path.name)
        return self._zip

    def get_audio_path(self, track_id: int) -> str:
        """
        Build the internal ZIP path for a track's MP3 file.

        Uses the configured template, track ID padding, and folder prefix
        from configs/paths.yaml.

        Args:
            track_id: Numeric track identifier.

        Returns:
            Relative path string within the ZIP archive.
        """
        track_str = str(track_id).zfill(paths.track_id_padding)
        folder = track_str[:paths.folder_chars]
        subset = paths.active_subset
        return paths.zip_template.format(
            subset=subset,
            folder=folder,
            track_id=track_str,
        )

    def _get_disk_cache_path(self, track_id: int, sr: int, duration: float) -> Path:
        """
        Generate a deterministic filesystem path for disk-cached audio.

        Uses an MD5 hash of the cache key to avoid filename length issues.

        Args:
            track_id: Track identifier.
            sr: Sample rate used for loading.
            duration: Duration in seconds used for loading.

        Returns:
            Path to the .npy cache file.
        """
        key = f"track_{track_id}_sr_{sr}_dur_{duration}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.disk_cache_dir / f"{hash_key}.npy"

    def _load_from_disk_cache(
        self, track_id: int, sr: int, duration: float
    ) -> Optional[np.ndarray]:
        """
        Attempt to load a waveform from the disk cache.

        Args:
            track_id: Track identifier.
            sr: Sample rate.
            duration: Duration in seconds.

        Returns:
            Numpy array if cached, None otherwise.
        """
        if not self.use_disk_cache:
            return None

        cache_path = self._get_disk_cache_path(track_id, sr, duration)
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                logger.debug(
                    "Corrupt disk cache for track %d, will reload", track_id
                )
        return None

    def _save_to_disk_cache(
        self, audio: np.ndarray, track_id: int, sr: int, duration: float
    ) -> None:
        """
        Persist a waveform to the disk cache.

        Args:
            audio: Waveform array to cache.
            track_id: Track identifier.
            sr: Sample rate.
            duration: Duration in seconds.
        """
        if not self.use_disk_cache:
            return

        cache_path = self._get_disk_cache_path(track_id, sr, duration)
        try:
            np.save(cache_path, audio)
        except Exception:
            logger.warning(
                "Failed to write disk cache for track %d: %s", track_id, cache_path
            )

    def load_audio_with_status(
        self,
        track_id: int,
        sr: Optional[int] = None,
        duration: Optional[float] = None,
        offset: float = 0.0,
        use_cache: bool = True,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Load audio for a track with detailed status reporting.

        Checks disk cache, then memory cache, then reads and decodes the
        MP3 from the ZIP archive. Resamples to the target sample rate,
        trims or pads to the exact target duration.

        Args:
            track_id: Numeric track identifier.
            sr: Target sample rate in Hz. Defaults to audio_params.sample_rate.
            duration: Target duration in seconds. Defaults to audio_params.duration.
            offset: Start offset in seconds for partial loading.
            use_cache: Whether to use and populate the in-memory cache.

        Returns:
            Tuple of (audio_array, status_dict).

            audio_array: 1D numpy float32 array, or None on failure.
            status_dict keys:
                - success: bool
                - error_type: str or None ('not_found', 'processing_error')
                - error_message: str or None
                - loaded_from_cache: bool
                - cache_type: 'disk', 'memory', or None
                - track_id: int
                - sr: int
                - duration: float
        """
        sr = sr or audio_params.sample_rate
        duration = duration or audio_params.duration

        status: Dict[str, Any] = {
            "success": False,
            "error_type": None,
            "error_message": None,
            "loaded_from_cache": False,
            "cache_type": None,
            "track_id": track_id,
            "sr": sr,
            "duration": duration,
        }

        audio = self._load_from_disk_cache(track_id, sr, duration)
        if audio is not None:
            status["success"] = True
            status["loaded_from_cache"] = True
            status["cache_type"] = "disk"
            return audio, status

        cache_key = (track_id, sr, duration, offset)
        if use_cache and self.use_cache and cache_key in self._cached_audio:
            status["success"] = True
            status["loaded_from_cache"] = True
            status["cache_type"] = "memory"
            return self._cached_audio[cache_key], status

        try:
            audio_path = self.get_audio_path(track_id)

            if audio_path not in self.zip.namelist():
                alt_path = f"fma_medium/{track_id:06d}.mp3"
                if alt_path in self.zip.namelist():
                    audio_path = alt_path
                else:
                    status["error_type"] = "not_found"
                    status["error_message"] = (
                        f"Track {track_id} not found in archive "
                        f"(tried: {audio_path}, {alt_path})"
                    )
                    logger.warning(
                        "Track %d not found in ZIP (tried primary and fallback)",
                        track_id,
                    )
                    return None, status

            with self.zip.open(audio_path) as f:
                audio_bytes = f.read()

            audio, _ = librosa.load(
                io.BytesIO(audio_bytes),
                sr=sr,
                duration=duration,
                offset=offset,
                res_type="kaiser_fast",
            )

            expected_length = int(sr * duration)
            if len(audio) < expected_length:
                audio = np.pad(audio, (0, expected_length - len(audio)))
            else:
                audio = audio[:expected_length]

            if self.use_disk_cache:
                self._save_to_disk_cache(audio, track_id, sr, duration)

            if (
                use_cache
                and self.use_cache
                and len(self._cached_audio) < self.cache_size
            ):
                self._cached_audio[cache_key] = audio

            status["success"] = True
            logger.debug("Track %d loaded successfully, shape=%s", track_id, audio.shape)
            return audio, status

        except KeyError as e:
            status["error_type"] = "not_found"
            status["error_message"] = (
                f"Track {track_id} not found in archive: {e}"
            )
            logger.warning("Track %d missing from ZIP: %s", track_id, e)
            return None, status

        except Exception as e:
            status["error_type"] = "processing_error"
            status["error_message"] = str(e)
            logger.warning(
                "Failed to load track %d: %s",
                track_id,
                e,
            )
            return None, status

    def load_audio(
        self,
        track_id: int,
        sr: Optional[int] = None,
        duration: Optional[float] = None,
        offset: float = 0.0,
        use_cache: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Load audio for a track (compatibility wrapper).

        Calls load_audio_with_status and returns only the array.
        Prefer load_audio_with_status for new code to handle errors explicitly.

        Args:
            track_id: Numeric track identifier.
            sr: Target sample rate. Defaults to audio_params.sample_rate.
            duration: Target duration. Defaults to audio_params.duration.
            offset: Start offset in seconds.
            use_cache: Whether to use the in-memory cache.

        Returns:
            1D numpy float32 array, or None on failure.
        """
        audio, _ = self.load_audio_with_status(
            track_id, sr, duration, offset, use_cache
        )
        return audio

    def load_audio_batch_with_status(
        self,
        track_ids: List[int],
        sr: Optional[int] = None,
        duration: Optional[float] = None,
        offset: float = 0.0,
    ) -> Tuple[Dict[int, Optional[np.ndarray]], Dict[int, Dict[str, Any]]]:
        """
        Load multiple tracks with per-track status reporting.

        Processes tracks sequentially. For large batches, progress is
        logged at DEBUG level every 100 tracks.

        Args:
            track_ids: List of track identifiers to load.
            sr: Target sample rate.
            duration: Target duration.
            offset: Start offset in seconds.

        Returns:
            Tuple of (audio_dict, status_dict).
            - audio_dict: {track_id: numpy_array_or_None}
            - status_dict: {track_id: status_dict}
        """
        audio_results: Dict[int, Optional[np.ndarray]] = {}
        status_results: Dict[int, Dict[str, Any]] = {}
        total = len(track_ids)
        failed_ids: List[int] = []

        for i, track_id in enumerate(track_ids):
            if total >= 200 and (i + 1) % 100 == 0:
                logger.debug(
                    "Batch progress: %d/%d tracks loaded (%d failed so far)",
                    i + 1,
                    total,
                    len(failed_ids),
                )

            audio, status = self.load_audio_with_status(track_id, sr, duration, offset)
            audio_results[track_id] = audio
            status_results[track_id] = status

            if not status["success"]:
                failed_ids.append(track_id)

        n_failed = len(failed_ids)
        if n_failed > 0:
            logger.warning(
                "Batch complete: %d/%d tracks failed to load%s",
                n_failed,
                total,
                f" — first 10: {failed_ids[:10]}" if n_failed > 0 else "",
            )
        else:
            logger.info("Batch complete: all %d tracks loaded successfully", total)

        return audio_results, status_results

    def get_available_tracks(self) -> List[int]:
        """
        Scan the ZIP archive for all available MP3 tracks.

        Parses filenames matching the configured template pattern
        for the active subset.

        Returns:
            Sorted list of integer track IDs found in the archive.
        """
        tracks: List[int] = []
        subset = paths.active_subset
        prefix = f"fma_{subset}/"

        for name in self.zip.namelist():
            if name.endswith(".mp3") and name.startswith(prefix):
                parts = name.replace(".mp3", "").split("/")
                if len(parts) >= 3:
                    try:
                        track_id = int(parts[-1])
                        tracks.append(track_id)
                    except ValueError:
                        pass

        logger.debug(
            "Found %d tracks in ZIP matching subset '%s'", len(tracks), subset
        )
        return sorted(tracks)

    def get_failed_tracks_report(
        self, status_results: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate error statistics from a batch status dictionary.

        Args:
            status_results: Per-track status dictionaries as returned by
                            load_audio_batch_with_status.

        Returns:
            Report dictionary with keys:
            - total_processed: int
            - failed_count: int
            - failed_tracks: list of track IDs
            - failed_by_type: dict mapping error_type to list of track IDs
            - success_rate: float (0.0 to 1.0)
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

    def close(self) -> None:
        """Close the ZIP archive handle if open."""
        if self._zip is not None:
            self._zip.close()
            self._zip = None
            logger.debug("ZIP archive closed")

    def clear_cache(self) -> None:
        """Clear the in-memory audio cache."""
        n_cleared = len(self._cached_audio)
        self._cached_audio.clear()
        logger.info("Memory cache cleared (%d tracks removed)", n_cleared)

    def clear_disk_cache(self) -> None:
        """Remove all files from the disk cache directory."""
        if not self.disk_cache_dir.exists():
            return

        file_count = len(list(self.disk_cache_dir.glob("*")))
        size_mb = (
            sum(f.stat().st_size for f in self.disk_cache_dir.glob("*"))
            / (1024 ** 2)
        )

        shutil.rmtree(self.disk_cache_dir)
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Disk cache cleared: %s (%d files, %.1f MB freed)",
            self.disk_cache_dir,
            file_count,
            size_mb,
        )

    def __del__(self) -> None:
        """Ensure ZIP is closed on garbage collection."""
        self.close()


if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    setup_logging(level=logging.DEBUG, mode="console")

    loader = AudioLoader()
    tracks = loader.get_available_tracks()
    print(f"\nAvailable tracks: {len(tracks)}")
    print(f"First 10: {tracks[:10]}")

    if tracks:
        audio, status = loader.load_audio_with_status(tracks[0])
        if status["success"]:
            print(f"Test track {tracks[0]}: shape={audio.shape}, "
                  f"min={audio.min():.3f}, max={audio.max():.3f}")

    loader.close()