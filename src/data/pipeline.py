"""
src/data/pipeline.py
Complete data preparation pipeline for FMA genre classification.

Orchestrates the full sequence from raw metadata to normalized, encoded,
split-ready datasets with automatic caching. The pipeline:

1. Loads track metadata from FMA CSV files
2. Filters tracks without genre labels
3. Removes genres with insufficient samples
4. Applies the official FMA train/validation/test split
5. Joins precomputed features from features.csv
6. Encodes genre labels to integer IDs
7. Normalizes feature vectors (StandardScaler)
8. Optionally verifies audio file availability in ZIP archives
9. Optionally filters out tracks with missing or corrupt audio
10. Caches all results for instant reloading

Typical usage:
    from src.data.pipeline import DataPipeline

    pipeline = DataPipeline(subset="medium", min_samples_per_genre=100)
    data = pipeline.run()

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    # With audio filtering (synchronized XGBoost + CNN datasets):
    pipeline = DataPipeline(
        subset="medium",
        min_samples_per_genre=100,
        filter_corrupt_audio=True,
    )
    data = pipeline.run()
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import paths
from src.data.loader import FMALoader
from src.data.preprocessor import DataPreprocessor


logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data preparation pipeline for FMA genre classification.

    Handles metadata loading, genre filtering, label encoding, feature
    normalization, train/val/test splitting, audio availability checking,
    optional removal of corrupt audio tracks, and transparent disk caching
    of all intermediate results.

    The official FMA split (80/10/10) is used. All preprocessing state
    (LabelEncoder, StandardScaler) is fitted exclusively on the training
    set to prevent data leakage.

    Attributes:
        subset: FMA subset identifier ('small', 'medium', 'large').
        min_samples_per_genre: Minimum track count to retain a genre.
        use_features: Whether to use precomputed features from features.csv.
        check_audio_availability: Whether to verify audio file presence in ZIP.
        filter_corrupt_audio: Whether to remove tracks with missing/corrupt audio.
        dataset_id: Unique string identifier for this pipeline configuration.
        loader: FMALoader instance for metadata access.
        preprocessor: DataPreprocessor instance for encoding and scaling.
    """

    def __init__(
        self,
        subset: str = "medium",
        min_samples_per_genre: int = 100,
        use_features: bool = True,
        check_audio_availability: bool = True,
        filter_corrupt_audio: bool = False,
    ):
        """
        Initialize the data pipeline.

        Args:
            subset: FMA subset to use. One of 'small' (8k tracks),
                    'medium' (25k tracks), or 'large' (106k tracks).
            min_samples_per_genre: Minimum number of tracks a genre must
                                   have to be retained in the dataset.
            use_features: If True, loads precomputed features from
                          features.csv (518 features). If False, uses
                          only basic track metadata.
            check_audio_availability: If True, verifies which tracks have
                                      corresponding audio files in the ZIP
                                      archive and generates a missing-track
                                      report.
            filter_corrupt_audio: If True, removes tracks with missing or
                                  corrupt audio files from all splits.
                                  Requires check_audio_availability=True
                                  (enabled automatically if needed).
                                  Produces a synchronized dataset for both
                                  XGBoost and CNN training.
        """
        self.subset = subset
        self.min_samples_per_genre = min_samples_per_genre
        self.use_features = use_features
        self.check_audio_availability = check_audio_availability
        self.filter_corrupt_audio = filter_corrupt_audio

        if filter_corrupt_audio and not check_audio_availability:
            logger.warning(
                "filter_corrupt_audio=True requires check_audio_availability=True. "
                "Enabling check_audio_availability automatically."
            )
            self.check_audio_availability = True

        self.dataset_id = (
            f"{subset}_{min_samples_per_genre}"
        )

        self._setup_paths()

        self.loader = FMALoader()
        self.preprocessor = DataPreprocessor(min_samples_per_genre)

        self._data: Optional[Dict[str, Any]] = None
        self._missing_audio_report: Optional[Dict[str, Any]] = None

        logger.info(
            "DataPipeline initialized — subset=%s, min_samples=%d, "
            "use_features=%s, check_audio=%s, filter_corrupt=%s, "
            "dataset_id=%s",
            subset,
            min_samples_per_genre,
            use_features,
            check_audio_availability,
            filter_corrupt_audio,
            self.dataset_id,
        )

    def _setup_paths(self) -> None:
        """Configure cache directories and file paths for this dataset."""
        self.data_dir = paths.fma_features_dataset_dir / self.dataset_id
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.processors_dir = paths.processors_data_dir / "fma"
        self.processors_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_cache_file = (
            self.processors_dir / f"pipeline_{self.dataset_id}.pkl"
        )
        self.preprocessor_file = (
            self.processors_dir / f"preprocessor_{self.dataset_id}.pkl"
        )
        self.metadata_file = self.data_dir / "metadata.json"
        self.missing_report_file = self.data_dir / "missing_audio_report.json"

        self.X_train_file = self.data_dir / "X_train.npy"
        self.X_val_file = self.data_dir / "X_val.npy"
        self.X_test_file = self.data_dir / "X_test.npy"
        self.y_train_file = self.data_dir / "y_train.npy"
        self.y_val_file = self.data_dir / "y_val.npy"
        self.y_test_file = self.data_dir / "y_test.npy"
        self.train_indices_file = self.data_dir / "train_indices.npy"
        self.val_indices_file = self.data_dir / "val_indices.npy"
        self.test_indices_file = self.data_dir / "test_indices.npy"

    def _is_cached(self) -> bool:
        """Check whether cached data exists for this dataset configuration."""
        return (
            self.dataset_cache_file.exists()
            and self.metadata_file.exists()
            and self.X_train_file.exists()
        )

    def run(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Execute the complete data preparation pipeline.

        On first run, processes all data from scratch and caches results.
        On subsequent runs, loads from cache unless force_reload is True.

        Args:
            force_reload: If True, ignores cached data and reprocesses.

        Returns:
            Dictionary containing:
            - X_train, X_val, X_test: normalized feature arrays
            - y_train, y_val, y_test: encoded label arrays
            - label_encoder: fitted LabelEncoder
            - scaler: fitted StandardScaler
            - genre_names: list of genre name strings
            - metadata: dataset statistics dictionary
            - train_indices, val_indices, test_indices: track ID arrays
            - audio_availability: per-split audio presence stats (if checked)
            - missing_audio_report: detailed missing track info (if checked)
        """
        if not force_reload and self._is_cached():
            logger.info("Loading data from cache [%s]", self.dataset_id)
            return self._load_from_cache()

        logger.info(
            "Starting full pipeline run [%s] — subset=%s, min_samples=%d",
            self.dataset_id,
            self.subset.upper(),
            self.min_samples_per_genre,
        )

        tracks = self.loader.get_tracks_by_subset(self.subset)

        genre_col = ("track", "genre_top")
        if genre_col not in tracks.columns:
            raise KeyError(
                f"Genre column {genre_col} not found. "
                f"Available: {list(tracks.columns)}"
            )

        tracks_with_genre = tracks[tracks[genre_col].notna()].copy()
        n_no_genre = len(tracks) - len(tracks_with_genre)
        if n_no_genre > 0:
            logger.info(
                "Removed %d tracks without genre label (%d remaining)",
                n_no_genre,
                len(tracks_with_genre),
            )

        tracks_filtered = self.preprocessor.filter_rare_genres(
            tracks_with_genre, genre_col
        )

        splits = self.loader.get_available_splits(tracks_filtered)

        train_idx = splits["training"]
        val_idx = splits["validation"]
        test_idx = splits["test"]

        n_total = len(tracks_filtered)
        logger.info(
            "Official FMA split — train: %d (%.1f%%), val: %d (%.1f%%), "
            "test: %d (%.1f%%)",
            len(train_idx),
            100 * len(train_idx) / n_total,
            len(val_idx),
            100 * len(val_idx) / n_total,
            len(test_idx),
            100 * len(test_idx) / n_total,
        )

        if self.use_features:
            features_all = self.loader.features

            common_idx = tracks_filtered.index.intersection(features_all.index)
            n_missing_features = len(tracks_filtered) - len(common_idx)
            if n_missing_features > 0:
                logger.warning(
                    "%d tracks (%.1f%%) have no features in features.csv "
                    "and will be excluded",
                    n_missing_features,
                    100 * n_missing_features / len(tracks_filtered),
                )

            X_all = features_all.loc[common_idx]
            y_all = tracks_filtered.loc[common_idx, genre_col]

            train_idx = common_idx.intersection(train_idx)
            val_idx = common_idx.intersection(val_idx)
            test_idx = common_idx.intersection(test_idx)
        else:
            logger.info("Using basic track metadata instead of features.csv")
            X_all = tracks_filtered[["track", "bit_rate"]].copy()
            y_all = tracks_filtered[genre_col]

        X_train_raw = X_all.loc[train_idx]
        X_val_raw = X_all.loc[val_idx]
        X_test_raw = X_all.loc[test_idx]

        y_train_raw = y_all.loc[train_idx]
        y_val_raw = y_all.loc[val_idx]
        y_test_raw = y_all.loc[test_idx]

        logger.debug(
            "Raw feature shapes — train: %s, val: %s, test: %s",
            X_train_raw.shape,
            X_val_raw.shape,
            X_test_raw.shape,
        )

        y_train, y_val, y_test = self.preprocessor.encode_labels(
            y_train_raw, y_val_raw, y_test_raw
        )

        X_train, X_val, X_test = self.preprocessor.normalize_features(
            X_train_raw, X_val_raw, X_test_raw
        )

        metadata = {
            "dataset_id": self.dataset_id,
            "subset": self.subset,
            "min_samples_per_genre": self.min_samples_per_genre,
            "num_samples": len(X_all),
            "num_features": X_all.shape[1],
            "num_classes": len(self.preprocessor.label_encoder.classes_),
            "class_names": list(self.preprocessor.label_encoder.classes_),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "use_features": self.use_features,
            "split_method": "official_fma_80_10_10",
            "feature_names": (
                list(X_all.columns)
                if isinstance(X_all, pd.DataFrame)
                else None
            ),
            "filtered_corrupt_audio": False,
            "created_at": pd.Timestamp.now().isoformat(),
        }

        self._data = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "label_encoder": self.preprocessor.label_encoder,
            "scaler": self.preprocessor.scaler,
            "genre_names": metadata["class_names"],
            "metadata": metadata,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
            "missing_audio_report": None,
        }

        if self.check_audio_availability:
            self._check_audio_availability(train_idx, val_idx, test_idx)

        if self.filter_corrupt_audio and self._data.get("audio_availability"):
            self._apply_audio_filter()

        self._save_to_cache()

        self._log_final_summary()

        return self._data

    def _check_audio_availability(
        self,
        train_idx: pd.Index,
        val_idx: pd.Index,
        test_idx: pd.Index,
    ) -> None:
        """
        Verify which tracks have corresponding audio files in the ZIP archive.

        Builds per-split availability statistics including lists of available
        and missing track IDs for optional filtering. Results are stored in
        self._data['audio_availability'] and self._missing_audio_report.

        Args:
            train_idx: Training set track ID indices.
            val_idx: Validation set track ID indices.
            test_idx: Test set track ID indices.
        """
        logger.info("Checking audio file availability in ZIP archive...")

        try:
            from src.data.audio_loader import AudioLoader

            audio_loader = AudioLoader(use_disk_cache=False)
            available_tracks = set(audio_loader.get_available_tracks())

            def analyze_availability(indices: pd.Index) -> Dict[str, Any]:
                idx_list = list(indices)
                available = [i for i in idx_list if i in available_tracks]
                missing = [i for i in idx_list if i not in available_tracks]
                return {
                    "total": len(idx_list),
                    "available_count": len(available),
                    "available_ids": available,
                    "missing_count": len(missing),
                    "missing_ids": missing,
                    "available_rate": (
                        len(available) / len(idx_list) if idx_list else 0.0
                    ),
                }

            train_analysis = analyze_availability(train_idx)
            val_analysis = analyze_availability(val_idx)
            test_analysis = analyze_availability(test_idx)

            self._data["audio_availability"] = {
                "train": train_analysis,
                "val": val_analysis,
                "test": test_analysis,
            }

            self._missing_audio_report = {
                "train_missing": [int(x) for x in train_analysis["missing_ids"]],
                "val_missing": [int(x) for x in val_analysis["missing_ids"]],
                "test_missing": [int(x) for x in test_analysis["missing_ids"]],
                "total_missing": (
                    train_analysis["missing_count"]
                    + val_analysis["missing_count"]
                    + test_analysis["missing_count"]
                ),
            }

            total_missing = self._missing_audio_report["total_missing"]
            if total_missing > 0:
                logger.warning(
                    "Audio availability check: %d tracks missing in ZIP "
                    "(train: %d, val: %d, test: %d)",
                    total_missing,
                    train_analysis["missing_count"],
                    val_analysis["missing_count"],
                    test_analysis["missing_count"],
                )
            else:
                logger.info("Audio availability check: all tracks present in ZIP")

            audio_loader.close()

        except Exception as e:
            logger.error("Failed to check audio availability: %s", e, exc_info=True)
            self._data["audio_availability"] = None

    def _apply_audio_filter(self) -> None:
        """
        Remove tracks whose audio is missing or corrupt from all splits.

        Updates X_train, y_train, etc. and all indices in self._data.
        Must be called after _check_audio_availability populated
        self._data['audio_availability'].
        """
        audio_avail = self._data.get("audio_availability")
        if audio_avail is None:
            logger.warning(
                "Audio availability data missing — cannot apply filter. "
                "Skipping."
            )
            return

        available_train = set(audio_avail["train"]["available_ids"])
        available_val = set(audio_avail["val"]["available_ids"])
        available_test = set(audio_avail["test"]["available_ids"])

        original_train = len(self._data["train_indices"])
        original_val = len(self._data["val_indices"])
        original_test = len(self._data["test_indices"])

        train_mask = np.isin(self._data["train_indices"], list(available_train))
        val_mask = np.isin(self._data["val_indices"], list(available_val))
        test_mask = np.isin(self._data["test_indices"], list(available_test))

        self._data["train_indices"] = self._data["train_indices"][train_mask]
        self._data["val_indices"] = self._data["val_indices"][val_mask]
        self._data["test_indices"] = self._data["test_indices"][test_mask]

        self._data["X_train"] = self._data["X_train"][train_mask]
        self._data["X_val"] = self._data["X_val"][val_mask]
        self._data["X_test"] = self._data["X_test"][test_mask]

        self._data["y_train"] = self._data["y_train"][train_mask]
        self._data["y_val"] = self._data["y_val"][val_mask]
        self._data["y_test"] = self._data["y_test"][test_mask]

        removed_train = original_train - len(self._data["train_indices"])
        removed_val = original_val - len(self._data["val_indices"])
        removed_test = original_test - len(self._data["test_indices"])
        total_original = original_train + original_val + original_test
        total_removed = removed_train + removed_val + removed_test

        logger.info(
            "Audio filter applied — removed %d train, %d val, %d test "
            "tracks (%.1f%% of total)",
            removed_train,
            removed_val,
            removed_test,
            100 * total_removed / total_original if total_original > 0 else 0.0,
        )

        self._data["metadata"]["train_size"] = len(self._data["X_train"])
        self._data["metadata"]["val_size"] = len(self._data["X_val"])
        self._data["metadata"]["test_size"] = len(self._data["X_test"])
        self._data["metadata"]["num_samples"] = (
            len(self._data["X_train"])
            + len(self._data["X_val"])
            + len(self._data["X_test"])
        )
        self._data["metadata"]["filtered_corrupt_audio"] = True
        self._data["metadata"]["removed_corrupt"] = {
            "train": removed_train,
            "val": removed_val,
            "test": removed_test,
        }

    def _save_to_cache(self) -> None:
        """
        Persist all prepared data to disk cache.

        Saves feature/label arrays as .npy files, metadata as JSON,
        the preprocessor as a joblib archive, and the full pipeline
        object as a pickle for quick reloading.
        """
        if self._data is None:
            return

        logger.info("Saving data to cache: %s", self.data_dir)

        with open(self.dataset_cache_file, "wb") as f:
            pickle.dump(self._data, f)

        np.save(self.X_train_file, self._data["X_train"])
        np.save(self.X_val_file, self._data["X_val"])
        np.save(self.X_test_file, self._data["X_test"])
        np.save(self.y_train_file, self._data["y_train"])
        np.save(self.y_val_file, self._data["y_val"])
        np.save(self.y_test_file, self._data["y_test"])

        if self._data.get("train_indices") is not None:
            np.save(self.train_indices_file, self._data["train_indices"])
            np.save(self.val_indices_file, self._data["val_indices"])
            np.save(self.test_indices_file, self._data["test_indices"])

        if self._missing_audio_report:
            with open(self.missing_report_file, "w") as f:
                json.dump(self._missing_audio_report, f, indent=2)

        self.preprocessor.save(self.preprocessor_file)

        with open(self.metadata_file, "w") as f:
            metadata_copy = self._data["metadata"].copy()
            metadata_copy["class_names"] = list(metadata_copy["class_names"])
            if (
                "feature_names" in metadata_copy
                and metadata_copy["feature_names"] is not None
            ):
                metadata_copy["feature_names"] = list(
                    metadata_copy["feature_names"]
                )
            json.dump(metadata_copy, f, indent=2)

        version_info = {
            "dataset_id": self.dataset_id,
            "subset": self.subset,
            "min_samples_per_genre": self.min_samples_per_genre,
            "use_features": self.use_features,
            "filter_corrupt_audio": self.filter_corrupt_audio,
            "files": {
                "X_train": str(self.X_train_file),
                "X_val": str(self.X_val_file),
                "X_test": str(self.X_test_file),
                "y_train": str(self.y_train_file),
                "y_val": str(self.y_val_file),
                "y_test": str(self.y_test_file),
                "preprocessor": str(self.preprocessor_file),
                "metadata": str(self.metadata_file),
            },
        }
        with open(self.data_dir / "dataset_info.json", "w") as f:
            json.dump(version_info, f, indent=2)

        logger.info("Cache saved successfully [%s]", self.dataset_id)

    def _load_from_cache(self) -> Dict[str, Any]:
        """
        Restore prepared data from disk cache.

        Loads the full pipeline object, preprocessor state, track indices,
        and missing audio report.

        Returns:
            The cached data dictionary with all keys from run().
        """
        logger.info("Loading from cache [%s]", self.dataset_id)

        with open(self.dataset_cache_file, "rb") as f:
            self._data = pickle.load(f)

        if self.train_indices_file.exists():
            self._data["train_indices"] = np.load(self.train_indices_file)
            self._data["val_indices"] = np.load(self.val_indices_file)
            self._data["test_indices"] = np.load(self.test_indices_file)

        self.preprocessor.load(self.preprocessor_file)

        self._data["label_encoder"] = self.preprocessor.label_encoder
        self._data["scaler"] = self.preprocessor.scaler

        if self.missing_report_file.exists():
            with open(self.missing_report_file, "r") as f:
                self._missing_audio_report = json.load(f)
                self._data["missing_audio_report"] = self._missing_audio_report

        meta = self._data["metadata"]
        filter_info = ""
        if meta.get("filtered_corrupt_audio"):
            removed = meta.get("removed_corrupt", {})
            filter_info = (
                f", audio-filtered: -{removed.get('train', 0)}/"
                f"-{removed.get('val', 0)}/-{removed.get('test', 0)}"
            )

        logger.info(
            "Cached dataset: FMA %s — %d tracks, %d genres, "
            "train/val/test = %d/%d/%d%s",
            meta["subset"].upper(),
            meta["num_samples"],
            meta["num_classes"],
            meta["train_size"],
            meta["val_size"],
            meta["test_size"],
            filter_info,
        )

        if (
            self._missing_audio_report
            and self._missing_audio_report.get("total_missing", 0) > 0
        ):
            logger.warning(
                "Cached report: %d tracks missing in ZIP",
                self._missing_audio_report["total_missing"],
            )

        return self._data

    def _log_final_summary(self) -> None:
        """Log a structured summary of the completed pipeline run."""
        meta = self._data["metadata"]

        logger.info("=" * 40 + " PIPELINE COMPLETE " + "=" * 40)
        logger.info("Dataset ID:    %s", self.dataset_id)
        logger.info("FMA subset:    %s", meta["subset"].upper())
        logger.info("Total tracks:  %d", meta["num_samples"])
        logger.info("Features:      %d", meta["num_features"])
        logger.info("Genres:        %d", meta["num_classes"])
        logger.info(
            "Split — train: %d (%.1f%%), val: %d (%.1f%%), test: %d (%.1f%%)",
            meta["train_size"],
            100 * meta["train_size"] / meta["num_samples"] if meta["num_samples"] else 0,
            meta["val_size"],
            100 * meta["val_size"] / meta["num_samples"] if meta["num_samples"] else 0,
            meta["test_size"],
            100 * meta["test_size"] / meta["num_samples"] if meta["num_samples"] else 0,
        )

        if meta.get("filtered_corrupt_audio"):
            removed = meta.get("removed_corrupt", {})
            logger.info(
                "Audio filter — removed %d train, %d val, %d test",
                removed.get("train", 0),
                removed.get("val", 0),
                removed.get("test", 0),
            )

        if self._data.get("audio_availability") and not meta.get("filtered_corrupt_audio"):
            audio = self._data["audio_availability"]
            logger.info(
                "Audio in ZIP — train: %d/%d, val: %d/%d, test: %d/%d",
                audio["train"]["available_count"],
                audio["train"]["total"],
                audio["val"]["available_count"],
                audio["val"]["total"],
                audio["test"]["available_count"],
                audio["test"]["total"],
            )

        logger.info("Data cached at:  %s", self.data_dir)
        logger.info("Preprocessor at: %s", self.processors_dir)
        logger.info("=" * 97)

    def get_class_weights(self) -> Dict[int, float]:
        """
        Compute balanced class weights for the training set.

        Returns:
            Dictionary mapping class ID to weight value.

        Raises:
            RuntimeError: If the pipeline has not been run yet.
        """
        if self._data is None:
            raise RuntimeError("Pipeline has not been run. Call run() first.")
        return self.preprocessor.get_class_weights(self._data["y_train"])

    def get_missing_audio_report(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the missing audio track report.

        Returns:
            Dictionary with keys 'train_missing', 'val_missing',
            'test_missing', 'total_missing', or None if audio
            availability was not checked.
        """
        return self._missing_audio_report

    def get_available_indices(self, split: str = "train") -> List[int]:
        """
        Get track indices that have corresponding audio files in the ZIP.

        Args:
            split: One of 'train', 'val', 'test'.

        Returns:
            List of track IDs available in the ZIP archive.

        Raises:
            RuntimeError: If pipeline hasn't been run or audio wasn't checked.
        """
        if self._data is None:
            raise RuntimeError("Pipeline has not been run. Call run() first.")

        audio_avail = self._data.get("audio_availability")
        if audio_avail is None:
            raise RuntimeError(
                "Audio availability information is not available. "
                "Set check_audio_availability=True."
            )

        split_map = {
            "train": audio_avail["train"]["available_ids"],
            "val": audio_avail["val"]["available_ids"],
            "test": audio_avail["test"]["available_ids"],
        }
        if split not in split_map:
            raise ValueError(
                f"Unknown split: '{split}'. Expected: 'train', 'val', 'test'."
            )

        return split_map[split]

    def print_info(self) -> None:
        """
        Print a human-readable summary of the prepared dataset.

        This is a manual debugging/exploration utility. Shows
        configuration, split sizes, genre list, and audio availability.
        """
        if self._data is None:
            print("Data not prepared yet. Call pipeline.run() first.")
            return

        meta = self._data["metadata"]
        print("=" * 60)
        print("PREPARED DATASET SUMMARY")
        print("=" * 60)
        print(f"Dataset ID:  {self.dataset_id}")
        print(f"FMA subset:  {meta['subset'].upper()}")
        print(f"Tracks:      {meta['num_samples']}")
        print(f"Features:    {meta['num_features']}")
        print(f"Genres:      {meta['num_classes']}")
        print(f"\nSplit:")
        print(f"  Train:     {meta['train_size']}")
        print(f"  Val:       {meta['val_size']}")
        print(f"  Test:      {meta['test_size']}")

        if meta.get("filtered_corrupt_audio"):
            removed = meta.get("removed_corrupt", {})
            print(f"\nAudio filter applied:")
            print(f"  Removed:   {removed.get('train', 0)} train, "
                  f"{removed.get('val', 0)} val, "
                  f"{removed.get('test', 0)} test")

        if self._data.get("audio_availability") and not meta.get("filtered_corrupt_audio"):
            audio = self._data["audio_availability"]
            print(f"\nAudio availability in ZIP:")
            print(
                f"  Train: {audio['train']['available_count']}/"
                f"{audio['train']['total']} "
                f"({audio['train']['available_rate']:.1%})"
            )
            print(
                f"  Val:   {audio['val']['available_count']}/"
                f"{audio['val']['total']} "
                f"({audio['val']['available_rate']:.1%})"
            )
            print(
                f"  Test:  {audio['test']['available_count']}/"
                f"{audio['test']['total']} "
                f"({audio['test']['available_rate']:.1%})"
            )

        print(f"\nGenres:")
        for i, genre in enumerate(meta["class_names"][:10]):
            print(f"  {i}: {genre}")
        if len(meta["class_names"]) > 10:
            print(f"  ... and {len(meta['class_names']) - 10} more")


def run_pipeline(
    subset: str = "medium",
    min_samples_per_genre: int = 100,
    force_reload: bool = False,
    check_audio: bool = True,
    filter_corrupt_audio: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function for one-line pipeline execution.

    Args:
        subset: FMA subset ('small', 'medium', 'large').
        min_samples_per_genre: Minimum tracks per genre.
        force_reload: If True, ignore cache and reprocess.
        check_audio: If True, verify audio file availability.
        filter_corrupt_audio: If True, remove tracks with missing audio.

    Returns:
        Data dictionary with X_train, y_train, etc.
    """
    pipeline = DataPipeline(
        subset=subset,
        min_samples_per_genre=min_samples_per_genre,
        check_audio_availability=check_audio,
        filter_corrupt_audio=filter_corrupt_audio,
    )
    return pipeline.run(force_reload=force_reload)


if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    setup_logging(level=logging.DEBUG, mode="console")

    data = run_pipeline(
        subset="medium",
        min_samples_per_genre=50,
        force_reload=True,
        filter_corrupt_audio=True,
    )

    print("\nAvailable keys in result:")
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"   {key}: {data[key].shape}")
        elif isinstance(data[key], dict):
            print(f"   {key}: dict with {len(data[key])} keys")
        else:
            print(f"   {key}: {type(data[key]).__name__}")