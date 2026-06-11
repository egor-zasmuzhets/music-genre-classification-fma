"""
Complete data preparation pipeline for FMA genre classification (V2).

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
10. Caches all results in V2 format (JSON + .npy, no pickle)

Stacking support (V2):
- Added get_common_indices() for synchronizing CNN and XGBoost data
- Added get_split_info() for detailed split information
- Added validate_split_consistency() for cross-model validation

Typical usage:
    from src.data.pipeline import DataPipeline

    pipeline = DataPipeline(subset="medium", min_samples_per_genre=100)
    data = pipeline.run()

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    # For stacking ensemble - get common indices
    common_indices = pipeline.get_common_indices()
    train_indices = common_indices['train']
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import paths
from src.data.loader import FMALoader
from src.data.preprocessor import DataPreprocessor


logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data preparation pipeline for FMA genre classification (V2).

    Handles metadata loading, genre filtering, label encoding, feature
    normalization, train/val/test splitting, audio availability checking,
    optional removal of corrupt audio tracks, and transparent disk caching.

    V2: Caches in structured format with JSON metadata and .npy arrays.
    No pickle/joblib used.

    Stacking: Provides methods to synchronize data across different model types.

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
    ) -> None:
        """
        Initialize the data pipeline.

        Args:
            subset: FMA subset ('small', 'medium', 'large').
            min_samples_per_genre: Minimum tracks per genre to retain.
            use_features: If True, loads precomputed features from features.csv.
            check_audio_availability: Verify audio file presence in ZIP.
            filter_corrupt_audio: Remove tracks with missing/corrupt audio.
        """
        self.subset = subset
        self.min_samples_per_genre = min_samples_per_genre
        self.use_features = use_features
        self.check_audio_availability = check_audio_availability
        self.filter_corrupt_audio = filter_corrupt_audio

        if filter_corrupt_audio and not check_audio_availability:
            logger.warning(
                "filter_corrupt_audio=True requires check_audio_availability=True. "
                "Enabling automatically."
            )
            self.check_audio_availability = True

        self.dataset_id = f"{subset}_{min_samples_per_genre}"

        self._setup_paths()

        self.loader = FMALoader()
        self.preprocessor = DataPreprocessor(min_samples_per_genre)

        self._data: Optional[Dict[str, Any]] = None
        self._missing_audio_report: Optional[Dict[str, Any]] = None

        logger.info(
            "DataPipeline initialized — subset=%s, min_samples=%d, "
            "use_features=%s, check_audio=%s, filter_corrupt=%s, dataset_id=%s",
            subset, min_samples_per_genre, use_features,
            check_audio_availability, filter_corrupt_audio, self.dataset_id
        )

    def _setup_paths(self) -> None:
        """Configure cache directories and file paths for V2 format."""
        self.data_dir = paths.fma_features_dataset_dir / self.dataset_id
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # V2 structured directories
        self.features_dir = self.data_dir / "features"
        self.labels_dir = self.data_dir / "labels"
        self.indices_dir = self.data_dir / "indices"
        self.preprocessor_dir = self.data_dir / "preprocessor"

        for d in [self.features_dir, self.labels_dir, self.indices_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Metadata files
        self.metadata_file = self.data_dir / "metadata.json"
        self.missing_report_file = self.data_dir / "missing_audio_report.json"
        self.manifest_file = self.data_dir / "manifest.json"

    def _is_cached(self) -> bool:
        """Check whether cached data exists for this dataset configuration (V2)."""
        required = [
            self.metadata_file.exists(),
            (self.features_dir / "X_train.npy").exists(),
            (self.labels_dir / "y_train.npy").exists(),
            self.preprocessor_dir.exists(),
        ]
        return all(required)

    def run(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Execute the complete data preparation pipeline.

        On first run, processes all data from scratch and caches results.
        On subsequent runs, loads from cache unless force_reload is True.

        Args:
            force_reload: If True, ignores cached data and reprocesses.

        Returns:
            Dictionary containing prepared data.
        """
        if not force_reload and self._is_cached():
            logger.info("Loading data from V2 cache [%s]", self.dataset_id)
            return self._load_from_cache()

        logger.info(
            "Starting full pipeline run [%s] — subset=%s, min_samples=%d",
            self.dataset_id, self.subset.upper(), self.min_samples_per_genre
        )

        tracks = self.loader.get_tracks_by_subset(self.subset)

        genre_col = ("track", "genre_top")
        if genre_col not in tracks.columns:
            raise KeyError(f"Genre column {genre_col} not found")

        tracks_with_genre = tracks[tracks[genre_col].notna()].copy()
        n_no_genre = len(tracks) - len(tracks_with_genre)
        if n_no_genre > 0:
            logger.info("Removed %d tracks without genre label (%d remaining)",
                       n_no_genre, len(tracks_with_genre))

        tracks_filtered = self.preprocessor.filter_rare_genres(tracks_with_genre, genre_col)

        splits = self.loader.get_available_splits(tracks_filtered)

        train_idx = splits["training"]
        val_idx = splits["validation"]
        test_idx = splits["test"]

        n_total = len(tracks_filtered)
        logger.info(
            "Official FMA split — train: %d (%.1f%%), val: %d (%.1f%%), test: %d (%.1f%%)",
            len(train_idx), 100 * len(train_idx) / n_total if n_total else 0,
            len(val_idx), 100 * len(val_idx) / n_total if n_total else 0,
            len(test_idx), 100 * len(test_idx) / n_total if n_total else 0
        )

        if self.use_features:
            features_all = self.loader.features

            common_idx = tracks_filtered.index.intersection(features_all.index)
            n_missing_features = len(tracks_filtered) - len(common_idx)
            if n_missing_features > 0:
                logger.warning(
                    "%d tracks (%.1f%%) have no features and will be excluded",
                    n_missing_features,
                    100 * n_missing_features / len(tracks_filtered) if len(tracks_filtered) else 0
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

        logger.debug("Raw feature shapes — train: %s, val: %s, test: %s",
                    X_train_raw.shape, X_val_raw.shape, X_test_raw.shape)

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
            "feature_names": list(X_all.columns) if isinstance(X_all, pd.DataFrame) else None,
            "filtered_corrupt_audio": False,
            "format_version": 2,
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
            "train_indices": train_idx.values if hasattr(train_idx, 'values') else train_idx,
            "val_indices": val_idx.values if hasattr(val_idx, 'values') else val_idx,
            "test_indices": test_idx.values if hasattr(test_idx, 'values') else test_idx,
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
        """Verify which tracks have corresponding audio files in the ZIP archive."""
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
                    "available_rate": len(available) / len(idx_list) if len(idx_list) else 0.0,
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
                "total_missing": train_analysis["missing_count"] + val_analysis["missing_count"] + test_analysis["missing_count"],
            }

            total_missing = self._missing_audio_report["total_missing"]
            if total_missing > 0:
                logger.warning(
                    "Audio availability: %d tracks missing (train: %d, val: %d, test: %d)",
                    total_missing, train_analysis["missing_count"],
                    val_analysis["missing_count"], test_analysis["missing_count"]
                )
            else:
                logger.info("Audio availability: all tracks present in ZIP")

            audio_loader.close()

        except Exception as e:
            logger.error("Failed to check audio availability: %s", e, exc_info=True)
            self._data["audio_availability"] = None

    def _apply_audio_filter(self) -> None:
        """Remove tracks whose audio is missing or corrupt from all splits."""
        audio_avail = self._data.get("audio_availability")
        if audio_avail is None:
            logger.warning("Audio availability data missing — skipping filter")
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
            "Audio filter applied — removed %d train, %d val, %d test (%.1f%% of total)",
            removed_train, removed_val, removed_test,
            100 * total_removed / total_original if total_original > 0 else 0.0
        )

        self._data["metadata"]["train_size"] = len(self._data["X_train"])
        self._data["metadata"]["val_size"] = len(self._data["X_val"])
        self._data["metadata"]["test_size"] = len(self._data["X_test"])
        self._data["metadata"]["num_samples"] = (
            len(self._data["X_train"]) + len(self._data["X_val"]) + len(self._data["X_test"])
        )
        self._data["metadata"]["filtered_corrupt_audio"] = True
        self._data["metadata"]["removed_corrupt"] = {
            "train": removed_train,
            "val": removed_val,
            "test": removed_test,
        }

    def _save_to_cache(self) -> None:
        """Persist all prepared data to disk cache in V2 format."""
        if self._data is None:
            return

        logger.info("Saving data to V2 cache: %s", self.data_dir)

        # Save features
        np.save(self.features_dir / "X_train.npy", self._data["X_train"])
        np.save(self.features_dir / "X_val.npy", self._data["X_val"])
        np.save(self.features_dir / "X_test.npy", self._data["X_test"])

        # Save labels
        np.save(self.labels_dir / "y_train.npy", self._data["y_train"])
        np.save(self.labels_dir / "y_val.npy", self._data["y_val"])
        np.save(self.labels_dir / "y_test.npy", self._data["y_test"])

        # Save indices as JSON
        train_indices = self._data["train_indices"].tolist() if hasattr(self._data["train_indices"], 'tolist') else list(self._data["train_indices"])
        val_indices = self._data["val_indices"].tolist() if hasattr(self._data["val_indices"], 'tolist') else list(self._data["val_indices"])
        test_indices = self._data["test_indices"].tolist() if hasattr(self._data["test_indices"], 'tolist') else list(self._data["test_indices"])

        with open(self.indices_dir / "train.json", "w") as f:
            json.dump(train_indices, f)
        with open(self.indices_dir / "val.json", "w") as f:
            json.dump(val_indices, f)
        with open(self.indices_dir / "test.json", "w") as f:
            json.dump(test_indices, f)

        # Save missing audio report if exists
        if self._missing_audio_report:
            with open(self.missing_report_file, "w") as f:
                json.dump(self._missing_audio_report, f, indent=2)

        # Save preprocessor (JSON format)
        self.preprocessor.save(self.preprocessor_dir)

        # Save metadata
        metadata_copy = self._data["metadata"].copy()
        if "class_names" in metadata_copy:
            metadata_copy["class_names"] = list(metadata_copy["class_names"])
        if metadata_copy.get("feature_names"):
            metadata_copy["feature_names"] = list(metadata_copy["feature_names"])

        with open(self.metadata_file, "w") as f:
            json.dump(metadata_copy, f, indent=2)

        # Create manifest
        self._create_manifest()

        logger.info("V2 cache saved successfully [%s]", self.dataset_id)

    def _create_manifest(self) -> None:
        """Create manifest file with file hashes for integrity verification."""
        import hashlib

        def hash_file(path: Path) -> str:
            if not path.exists():
                return None
            sha256 = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()[:16]

        manifest = {
            "version": 2,
            "dataset_id": self.dataset_id,
            "created_at": pd.Timestamp.now().isoformat(),
            "files": {}
        }

        for f in self.features_dir.glob("*.npy"):
            manifest["files"][f"features/{f.name}"] = hash_file(f)
        for f in self.labels_dir.glob("*.npy"):
            manifest["files"][f"labels/{f.name}"] = hash_file(f)
        for f in self.indices_dir.glob("*.json"):
            manifest["files"][f"indices/{f.name}"] = hash_file(f)
        for f in self.preprocessor_dir.glob("*.json"):
            manifest["files"][f"preprocessor/{f.name}"] = hash_file(f)

        manifest["files"]["metadata.json"] = hash_file(self.metadata_file)

        with open(self.manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_from_cache(self) -> Dict[str, Any]:
        """Restore prepared data from V2 disk cache."""
        logger.info("Loading from V2 cache [%s]", self.dataset_id)

        # Load features
        X_train = np.load(self.features_dir / "X_train.npy")
        X_val = np.load(self.features_dir / "X_val.npy")
        X_test = np.load(self.features_dir / "X_test.npy")

        # Load labels
        y_train = np.load(self.labels_dir / "y_train.npy")
        y_val = np.load(self.labels_dir / "y_val.npy")
        y_test = np.load(self.labels_dir / "y_test.npy")

        # Load indices from JSON
        with open(self.indices_dir / "train.json", "r") as f:
            train_indices = np.array(json.load(f))
        with open(self.indices_dir / "val.json", "r") as f:
            val_indices = np.array(json.load(f))
        with open(self.indices_dir / "test.json", "r") as f:
            test_indices = np.array(json.load(f))

        # Load metadata
        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)

        # Load preprocessor
        self.preprocessor.load(self.preprocessor_dir)

        # Load missing audio report if exists
        if self.missing_report_file.exists():
            with open(self.missing_report_file, "r") as f:
                self._missing_audio_report = json.load(f)

        self._data = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "label_encoder": self.preprocessor.label_encoder,
            "scaler": self.preprocessor.scaler,
            "genre_names": metadata.get("class_names", []),
            "metadata": metadata,
            "dataset_id": self.dataset_id,
            "missing_audio_report": self._missing_audio_report,
        }

        filter_info = ""
        if metadata.get("filtered_corrupt_audio"):
            removed = metadata.get("removed_corrupt", {})
            filter_info = f", audio-filtered: -{removed.get('train', 0)}/-{removed.get('val', 0)}/-{removed.get('test', 0)}"

        logger.info(
            "Loaded V2 dataset: FMA %s — %d tracks, %d genres, train/val/test = %d/%d/%d%s",
            metadata["subset"].upper(), metadata["num_samples"], metadata["num_classes"],
            metadata["train_size"], metadata["val_size"], metadata["test_size"], filter_info
        )

        return self._data

    def get_common_indices(self) -> Dict[str, np.ndarray]:
        """
        Get track indices that exist in both CNN and XGBoost datasets.

        Ensures both models are trained on exactly the same tracks.
        This is critical for stacking ensemble to work correctly.

        If audio availability was checked, filters to tracks that actually
        have audio files in the ZIP archive.

        Returns:
            Dictionary with 'train', 'val', 'test' keys containing numpy arrays
            of track IDs that are available for both model types.

        Raises:
            RuntimeError: If pipeline has not been run yet.
        """
        if self._data is None:
            raise RuntimeError("Pipeline has not been run. Call run() first.")

        indices = {
            'train': self._data['train_indices'].copy(),
            'val': self._data['val_indices'].copy(),
            'test': self._data['test_indices'].copy()
        }

        # Filter by audio availability if checked
        if self._data.get('audio_availability'):
            audio_avail = self._data['audio_availability']
            for split in ['train', 'val', 'test']:
                available = set(audio_avail[split]['available_ids'])
                filtered = [int(i) for i in indices[split] if int(i) in available]
                indices[split] = np.array(filtered)
                logger.debug(f"  {split}: {len(indices[split])} tracks (audio available)")

        logger.info(
            "Common indices for stacking: train=%d, val=%d, test=%d",
            len(indices['train']), len(indices['val']), len(indices['test'])
        )

        return indices

    def get_split_info(self, split: str = "train") -> Dict[str, Any]:
        """
        Get detailed information about a specific split.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            Dictionary with split information:
            - indices: track IDs in this split
            - size: number of tracks
            - class_distribution: count per class
            - has_audio_available: whether audio availability was checked

        Raises:
            RuntimeError: If pipeline has not been run yet.
            ValueError: If split name is invalid.
        """
        if self._data is None:
            raise RuntimeError("Pipeline has not been run. Call run() first.")

        valid_splits = ['train', 'val', 'test']
        if split not in valid_splits:
            raise ValueError(f"Unknown split: '{split}'. Expected: {valid_splits}")

        indices = self._data[f'{split}_indices']

        # Get labels for these indices
        y = self._data[f'y_{split}']

        # Calculate class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = {
            int(self._data['genre_names'][i]): int(count)
            for i, count in zip(unique, counts)
        }

        result = {
            'split': split,
            'size': len(indices),
            'indices': indices.tolist(),
            'class_distribution': class_distribution,
            'has_audio_available': self._data.get('audio_availability') is not None
        }

        # Add audio availability if checked
        if self._data.get('audio_availability'):
            audio = self._data['audio_availability'][split]
            result['audio_available_count'] = audio['available_count']
            result['audio_missing_count'] = audio['missing_count']

        return result

    def validate_split_consistency(
        self,
        other_pipeline: 'DataPipeline'
    ) -> Dict[str, bool]:
        """
        Validate that splits are consistent between two pipeline instances.

        This is useful when comparing CNN and XGBoost datasets to ensure
        they can be used together in an ensemble.

        Args:
            other_pipeline: Another DataPipeline instance (e.g., for CNN features)

        Returns:
            Dictionary with consistency check results:
            - same_dataset_id: whether dataset_ids match
            - same_num_classes: whether number of classes matches
            - same_class_names: whether class names match
            - same_split_sizes: whether train/val/test sizes match
            - indices_overlap_train: overlap ratio for train indices
            - indices_overlap_val: overlap ratio for val indices
            - indices_overlap_test: overlap ratio for test indices

        Raises:
            RuntimeError: If either pipeline has not been run.
        """
        if self._data is None or other_pipeline._data is None:
            raise RuntimeError("Both pipelines must be run before validation")

        result = {
            'same_dataset_id': self.dataset_id == other_pipeline.dataset_id,
            'same_num_classes': self._data['metadata']['num_classes'] == other_pipeline._data['metadata']['num_classes'],
            'same_class_names': self._data['genre_names'] == other_pipeline._data['genre_names'],
            'same_split_sizes': (
                len(self._data['train_indices']) == len(other_pipeline._data['train_indices']) and
                len(self._data['val_indices']) == len(other_pipeline._data['val_indices']) and
                len(self._data['test_indices']) == len(other_pipeline._data['test_indices'])
            )
        }

        # Calculate overlap ratios
        for split in ['train', 'val', 'test']:
            self_set = set(self._data[f'{split}_indices'])
            other_set = set(other_pipeline._data[f'{split}_indices'])

            if len(self_set) > 0:
                overlap = len(self_set & other_set) / len(self_set)
            else:
                overlap = 0.0

            result[f'indices_overlap_{split}'] = overlap

        logger.info(
            "Split consistency: dataset_id_match=%s, class_match=%s, "
            "train_overlap=%.2f, val_overlap=%.2f, test_overlap=%.2f",
            result['same_dataset_id'], result['same_class_names'],
            result['indices_overlap_train'], result['indices_overlap_val'],
            result['indices_overlap_test']
        )

        return result

    def _log_final_summary(self) -> None:
        """Log a structured summary of the completed pipeline run."""
        meta = self._data["metadata"]

        logger.info("=" * 40 + " PIPELINE COMPLETE " + "=" * 40)
        logger.info("Dataset ID:    %s", self.dataset_id)
        logger.info("FMA subset:    %s", meta["subset"].upper())
        logger.info("Total tracks:  %d", meta["num_samples"])
        logger.info("Features:      %d", meta["num_features"])
        logger.info("Genres:        %d", meta["num_classes"])

        n_samples = meta["num_samples"]
        logger.info(
            "Split — train: %d (%.1f%%), val: %d (%.1f%%), test: %d (%.1f%%)",
            meta["train_size"], 100 * meta["train_size"] / n_samples if n_samples else 0,
            meta["val_size"], 100 * meta["val_size"] / n_samples if n_samples else 0,
            meta["test_size"], 100 * meta["test_size"] / n_samples if n_samples else 0
        )

        if meta.get("filtered_corrupt_audio"):
            removed = meta.get("removed_corrupt", {})
            logger.info("Audio filter — removed %d train, %d val, %d test",
                       removed.get("train", 0), removed.get("val", 0), removed.get("test", 0))

        if self._data.get("audio_availability") and not meta.get("filtered_corrupt_audio"):
            audio = self._data["audio_availability"]
            logger.info(
                "Audio in ZIP — train: %d/%d, val: %d/%d, test: %d/%d",
                audio["train"]["available_count"], audio["train"]["total"],
                audio["val"]["available_count"], audio["val"]["total"],
                audio["test"]["available_count"], audio["test"]["total"]
            )

        logger.info("Data cached at:  %s", self.data_dir)
        logger.info("=" * 97)

    def get_class_weights(self) -> Dict[int, float]:
        """Compute balanced class weights for the training set."""
        if self._data is None:
            raise RuntimeError("Pipeline has not been run. Call run() first.")
        return self.preprocessor.get_class_weights(self._data["y_train"])

    def get_missing_audio_report(self) -> Optional[Dict[str, Any]]:
        """Retrieve the missing audio track report."""
        return self._missing_audio_report

    def get_available_indices(self, split: str = "train") -> List[int]:
        """Get track indices that have corresponding audio files in the ZIP."""
        if self._data is None:
            raise RuntimeError("Pipeline has not been run. Call run() first.")

        audio_avail = self._data.get("audio_availability")
        if audio_avail is None:
            raise RuntimeError("Audio availability not available. Set check_audio_availability=True.")

        split_map = {
            "train": audio_avail["train"]["available_ids"],
            "val": audio_avail["val"]["available_ids"],
            "test": audio_avail["test"]["available_ids"],
        }
        if split not in split_map:
            raise ValueError(f"Unknown split: '{split}'. Expected: 'train', 'val', 'test'.")

        return split_map[split]

    def print_info(self) -> None:
        """Print a human-readable summary of the prepared dataset."""
        if self._data is None:
            print("Data not prepared yet. Call pipeline.run() first.")
            return

        meta = self._data["metadata"]
        print("=" * 60)
        print("PREPARED DATASET SUMMARY (V2)")
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
            print(f"  Removed:   {removed.get('train', 0)} train, {removed.get('val', 0)} val, {removed.get('test', 0)} test")

        if self._data.get("audio_availability") and not meta.get("filtered_corrupt_audio"):
            audio = self._data["audio_availability"]
            print(f"\nAudio availability in ZIP:")
            print(f"  Train: {audio['train']['available_count']}/{audio['train']['total']} ({audio['train']['available_rate']:.1%})")
            print(f"  Val:   {audio['val']['available_count']}/{audio['val']['total']} ({audio['val']['available_rate']:.1%})")
            print(f"  Test:  {audio['test']['available_count']}/{audio['test']['total']} ({audio['test']['available_rate']:.1%})")

        print(f"\nGenres:")
        for i, genre in enumerate(meta["class_names"][:10]):
            print(f"  {i}: {genre}")
        if len(meta["class_names"]) > 10:
            print(f"  ... and {len(meta['class_names']) - 10} more")

        print("-" * 60)
        print("Stacking support methods:")
        print("  - get_common_indices()   → synchronized indices for ensemble")
        print("  - get_split_info()       → detailed split information")
        print("  - validate_consistency() → cross-model validation")
        print("=" * 60)


def run_pipeline(
    subset: str = "medium",
    min_samples_per_genre: int = 100,
    force_reload: bool = False,
    check_audio: bool = True,
    filter_corrupt_audio: bool = False,
) -> Dict[str, Any]:
    """Convenience function for one-line pipeline execution."""
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
        min_samples_per_genre=100,
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