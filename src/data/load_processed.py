"""
Fast loading of preprocessed data from cache (V2).

Provides convenient access to datasets that have already been prepared
by the DataPipeline. Supports multiple dataset configurations identified
by subset and minimum samples per genre.

V2 Changes:
- Reads new directory structure with features/, labels/, indices/
- Loads preprocessor from JSON (not joblib)
- No legacy pickle/joblib support

Typical usage:
    from src.data.load_processed import load_data

    data = load_data()  # Uses default config (medium, 10)
    X_train, y_train = data['X_train'], data['y_train']

    # List available cached datasets
    from src.data.load_processed import list_datasets
    print(list_datasets())
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.config import paths
from src.data.preprocessor import DataPreprocessor


logger = logging.getLogger(__name__)


class LoadProcessedData:
    """
    Fast accessor for preprocessed and cached FMA datasets (V2 format).

    Loads feature arrays from .npy files in features/ directory,
    labels from labels/ directory, indices from JSON files,
    and preprocessor from JSON directory structure.

    Directory structure expected:
        data_dir/
        ├── features/
        │   ├── X_train.npy
        │   ├── X_val.npy
        │   └── X_test.npy
        ├── labels/
        │   ├── y_train.npy
        │   ├── y_val.npy
        │   └── y_test.npy
        ├── indices/
        │   ├── train.json
        │   ├── val.json
        │   └── test.json
        ├── preprocessor/
        │   ├── scaler.json
        │   ├── label_encoder.json
        │   └── config.json
        └── metadata.json

    Attributes:
        dataset_id: Unique identifier string for the dataset configuration.
        subset: FMA subset name ('small', 'medium', 'large').
        min_samples_per_genre: Minimum track count per genre used.
        data_dir: Path to the cached data directory.
    """

    def __init__(
        self,
        subset: Optional[str] = None,
        min_samples_per_genre: Optional[int] = None,
        dataset_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the data loader for a specific dataset configuration.

        Args:
            subset: FMA subset name. If None, uses configured default.
            min_samples_per_genre: Minimum tracks per genre. If None, uses 10.
            dataset_id: Full dataset identifier (e.g., 'medium_10').
                        Overrides subset and min_samples_per_genre if provided.
        """
        if dataset_id is not None:
            self.dataset_id = dataset_id
            parts = dataset_id.split("_")
            if len(parts) >= 2:
                self.subset = parts[0]
                try:
                    self.min_samples_per_genre = int(parts[1])
                except ValueError:
                    self.min_samples_per_genre = 10
            else:
                self.subset = subset or paths.active_subset
                self.min_samples_per_genre = min_samples_per_genre or 10
        else:
            self.subset = subset or paths.active_subset
            self.min_samples_per_genre = min_samples_per_genre or 10
            self.dataset_id = f"{self.subset}_{self.min_samples_per_genre}"

        self.data_dir = paths.fma_features_dataset_dir / self.dataset_id
        self._data: Optional[Dict[str, Any]] = None
        self._indices_cache: Dict[str, Dict[int, int]] = {}

        logger.debug(
            "LoadProcessedData initialized — dataset_id=%s, data_dir=%s",
            self.dataset_id, self.data_dir
        )

    def _get_features_path(self, split: str) -> Path:
        """Path to features .npy file for given split."""
        return self.data_dir / "features" / f"X_{split}.npy"

    def _get_labels_path(self, split: str) -> Path:
        """Path to labels .npy file for given split."""
        return self.data_dir / "labels" / f"y_{split}.npy"

    def _get_indices_path(self, split: str) -> Path:
        """Path to indices JSON file for given split."""
        return self.data_dir / "indices" / f"{split}.json"

    def _get_metadata_path(self) -> Path:
        """Path to the dataset metadata JSON file."""
        return self.data_dir / "metadata.json"

    def _get_preprocessor_dir(self) -> Path:
        """Path to preprocessor directory."""
        return self.data_dir / "preprocessor"

    def exists(self) -> bool:
        """
        Check whether cached data exists for this dataset configuration (V2 format).

        Returns:
            True if the cache directory contains all required files.
        """
        required = [
            self._get_features_path("train"),
            self._get_labels_path("train"),
            self._get_metadata_path(),
            self._get_preprocessor_dir().exists(),
        ]
        has_files = all(required)
        if not has_files:
            logger.debug("V2 cache not found for dataset '%s'", self.dataset_id)
        return has_files

    def list_available_datasets(self) -> List[str]:
        """
        Discover all cached dataset configurations on disk (V2 format).

        Scans for subdirectories containing a valid metadata.json file
        and the V2 directory structure.

        Returns:
            Sorted list of dataset_id strings.
        """
        if not paths.fma_features_dataset_dir.exists():
            logger.debug("Dataset directory not found: %s", paths.fma_features_dataset_dir)
            return []

        datasets = []
        for d in paths.fma_features_dataset_dir.iterdir():
            if not d.is_dir():
                continue
            # Check for V2 structure
            if (d / "metadata.json").exists() and (d / "features").exists():
                datasets.append(d.name)

        logger.debug("Found %d cached dataset(s): %s", len(datasets), datasets)
        return sorted(datasets)

    def load(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load preprocessed data from the disk cache (V2 format).

        Reads .npy arrays for features and labels, JSON for indices,
        and JSON-based preprocessor.

        Args:
            force_reload: If True, bypasses the in-memory cache.

        Returns:
            Data dictionary with keys:
            - X_train, X_val, X_test: normalized feature arrays
            - y_train, y_val, y_test: encoded label arrays
            - train_indices, val_indices, test_indices: track ID arrays
            - label_encoder: fitted LabelEncoder
            - scaler: fitted StandardScaler
            - genre_names: list of genre name strings
            - metadata: full dataset metadata dictionary
            - dataset_id: the dataset identifier string

        Raises:
            FileNotFoundError: If no cached data exists for this dataset_id.
        """
        if not self.exists():
            available = self.list_available_datasets()
            raise FileNotFoundError(
                f"Cached data not found for dataset '{self.dataset_id}'.\n"
                f"Searched in: {self.data_dir}\n"
                f"Available datasets: {available if available else 'none'}"
            )

        if self._data is not None and not force_reload:
            logger.debug("Returning in-memory cached data [%s]", self.dataset_id)
            return self._data

        logger.info("Loading cached data (V2) [%s] from: %s", self.dataset_id, self.data_dir)

        # Load features
        X_train = np.load(self._get_features_path("train"))
        X_val = np.load(self._get_features_path("val"))
        X_test = np.load(self._get_features_path("test"))

        # Load labels
        y_train = np.load(self._get_labels_path("train"))
        y_val = np.load(self._get_labels_path("val"))
        y_test = np.load(self._get_labels_path("test"))

        # Load indices from JSON
        with open(self._get_indices_path("train"), "r") as f:
            train_indices = np.array(json.load(f))
        with open(self._get_indices_path("val"), "r") as f:
            val_indices = np.array(json.load(f))
        with open(self._get_indices_path("test"), "r") as f:
            test_indices = np.array(json.load(f))

        # Load metadata
        with open(self._get_metadata_path(), "r") as f:
            metadata = json.load(f)

        # Load preprocessor from JSON directory
        preprocessor = DataPreprocessor()
        preprocessor.load(self._get_preprocessor_dir())

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
            "label_encoder": preprocessor.label_encoder,
            "scaler": preprocessor.scaler,
            "genre_names": metadata.get("class_names", []),
            "metadata": metadata,
            "dataset_id": self.dataset_id,
        }

        logger.info(
            "Loaded dataset [%s]: %d train, %d val, %d test — %d genres, %d features",
            self.dataset_id,
            len(X_train), len(X_val), len(X_test),
            metadata.get("num_classes", 0),
            metadata.get("num_features", 0),
        )

        return self._data

    def load_to_dataframe(self) -> Dict[str, Any]:
        """
        Load data and reconstruct pandas DataFrames with column names.

        If feature names are available in metadata, the feature arrays
        are wrapped in DataFrames with the original column labels.

        Returns:
            Data dictionary where X_train, X_val, X_test may be
            pandas DataFrames instead of numpy arrays.
        """
        data = self.load()
        feature_names = data["metadata"].get("feature_names")

        if feature_names:
            logger.debug("Reconstructing DataFrames with %d feature names", len(feature_names))
            return {
                **data,
                "X_train": pd.DataFrame(data["X_train"], columns=feature_names),
                "X_val": pd.DataFrame(data["X_val"], columns=feature_names),
                "X_test": pd.DataFrame(data["X_test"], columns=feature_names),
            }

        logger.debug("No feature names in metadata — returning raw numpy arrays")
        return data

    def get_indices_map(self, split: str = "train") -> Dict[int, int]:
        """
        Get mapping from track_id to position index for a specific split.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            Dictionary mapping track_id -> position in the array
        """
        valid_splits = ['train', 'val', 'test']
        if split not in valid_splits:
            raise ValueError(f"Unknown split: '{split}'. Expected: {valid_splits}")

        if split in self._indices_cache:
            return self._indices_cache[split]

        data = self.load()
        indices = data[f'{split}_indices']

        indices_map = {int(idx): i for i, idx in enumerate(indices)}
        self._indices_cache[split] = indices_map

        logger.debug(f"Built indices map for {split}: {len(indices_map)} entries")
        return indices_map

    def get_filtered_by_indices(
            self,
            track_ids: Union[List[int], np.ndarray],
            split: str = "train",
            return_indices: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Load data filtered by specific track indices.

        Args:
            track_ids: List or array of track IDs to filter
            split: One of 'train', 'val', 'test'
            return_indices: If True, also return the actual track IDs that were found

        Returns:
            Dictionary with:
            - 'X': filtered feature array
            - 'y': filtered labels
            - (if return_indices): 'indices' array of track IDs that were found
        """
        data = self.load()
        indices_map = self.get_indices_map(split)

        found_positions = []
        found_indices = []

        for tid in track_ids:
            tid_int = int(tid)
            if tid_int in indices_map:
                found_positions.append(indices_map[tid_int])
                found_indices.append(tid_int)

        if not found_positions:
            available = list(indices_map.keys())[:10]
            raise ValueError(
                f"No matching indices found in {split} split.\n"
                f"Requested {len(track_ids)} tracks, found 0.\n"
                f"Sample available tracks: {available}"
            )

        X = data[f'X_{split}'][found_positions]
        y = data[f'y_{split}'][found_positions]

        result = {'X': X, 'y': y}

        if return_indices:
            result['indices'] = np.array(found_indices)

        missing_count = len(track_ids) - len(found_positions)
        if missing_count > 0:
            logger.warning(
                f"Filtered {split}: found {len(found_positions)}/{len(track_ids)} tracks "
                f"({missing_count} missing)"
            )
        else:
            logger.debug(f"Filtered {split}: all {len(found_positions)} tracks found")

        return result

    def get_split_summary(self, split: str = "train") -> Dict[str, Any]:
        """
        Get a quick summary of a split without loading full data.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            Dictionary with split summary
        """
        data = self.load()
        y = data[f'y_{split}']
        genre_names = data['genre_names']

        unique, counts = np.unique(y, return_counts=True)

        class_distribution = {
            genre_names[i]: int(count) for i, count in zip(unique, counts)
        }

        return {
            'split': split,
            'size': len(y),
            'class_distribution': class_distribution,
            'class_names': genre_names,
            'min_samples_per_class': int(counts.min()) if len(counts) > 0 else 0,
            'max_samples_per_class': int(counts.max()) if len(counts) > 0 else 0,
            'num_classes': len(unique),
        }

    def clear_cache(self) -> None:
        """Clear in-memory cache and indices map."""
        self._data = None
        self._indices_cache = {}
        logger.debug("Cleared in-memory cache for %s", self.dataset_id)

    def print_info(self) -> None:
        """Print a human-readable summary of the cached dataset."""
        if not self.exists():
            print(f"Dataset not found: '{self.dataset_id}'")
            available = self.list_available_datasets()
            print(f"Available datasets: {available if available else 'none'}")
            return

        with open(self._get_metadata_path(), "r") as f:
            meta = json.load(f)

        print("=" * 60)
        print("CACHED DATASET INFO (V2)")
        print("=" * 60)
        print(f"Dataset ID:     {self.dataset_id}")
        print(f"FMA subset:     {meta.get('subset', '?').upper()}")
        print(f"Tracks:         {meta.get('num_samples', '?')}")
        print(f"Features:       {meta.get('num_features', '?')}")
        print(f"Genres:         {meta.get('num_classes', '?')}")
        print(f"\nSplit:")
        print(f"  Train:        {meta.get('train_size', '?')}")
        print(f"  Val:          {meta.get('val_size', '?')}")
        print(f"  Test:         {meta.get('test_size', '?')}")
        print(f"\nPaths:")
        print(f"  Data:         {self.data_dir}")
        print(f"  Preprocessor: {self._get_preprocessor_dir()}")


def load_data(
    subset: Optional[str] = None,
    min_samples_per_genre: int = 10,
    dataset_id: Optional[str] = None,
    as_dataframe: bool = False,
) -> Dict[str, Any]:
    """
    One-liner to load preprocessed FMA data from cache (V2 format).

    Args:
        subset: FMA subset ('small', 'medium', 'large').
        min_samples_per_genre: Minimum tracks per genre (default: 10).
        dataset_id: Full dataset ID string. Overrides subset/min if provided.
        as_dataframe: If True, returns feature data as pandas DataFrames.

    Returns:
        Data dictionary with X_train, y_train, etc.
    """
    loader = LoadProcessedData(
        subset=subset,
        min_samples_per_genre=min_samples_per_genre,
        dataset_id=dataset_id,
    )
    if as_dataframe:
        return loader.load_to_dataframe()
    return loader.load()


def load_track_indices(
    subset: Optional[str] = None,
    min_samples_per_genre: int = 10,
    dataset_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quickly load only the train/val/test track index arrays.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    loader = LoadProcessedData(
        subset=subset,
        min_samples_per_genre=min_samples_per_genre,
        dataset_id=dataset_id,
    )
    data = loader.load()
    return data["train_indices"], data["val_indices"], data["test_indices"]


def list_datasets() -> List[str]:
    """
    List all cached dataset configurations available on disk (V2 format).

    Returns:
        Sorted list of dataset_id strings.
    """
    loader = LoadProcessedData()
    return loader.list_available_datasets()


load = load_data
list_ds = list_datasets


if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    setup_logging(level=logging.DEBUG, mode="console")

    print("Available datasets:", list_datasets())

    try:
        data = load_data(subset="medium", min_samples_per_genre=10)
        print(f"\nLoaded dataset: {data['dataset_id']}")
        print(f"X_train shape: {data['X_train'].shape}")
        print(f"Genre names: {data['genre_names'][:5]}...")
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nRun DataPipeline first to create the dataset.")