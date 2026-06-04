"""
src/data/load_processed.py
Fast loading of preprocessed data from cache.

Provides convenient access to datasets that have already been prepared
by the DataPipeline. Supports multiple dataset configurations identified
by subset and minimum samples per genre.

Typical usage:
    from src.data.load_processed import load_data

    data = load_data()  # Uses default config (medium, 100)
    X_train, y_train = data['X_train'], data['y_train']

    # List available cached datasets
    from src.data.load_processed import list_datasets
    print(list_datasets())
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import paths


logger = logging.getLogger(__name__)


class LoadProcessedData:
    """
    Fast accessor for preprocessed and cached FMA datasets.

    Loads feature arrays, labels, metadata, preprocessor state,
    and track indices from the disk cache populated by DataPipeline.
    Supports discovery of multiple cached dataset configurations.

    Attributes:
        dataset_id: Unique identifier string for the dataset configuration.
        subset: FMA subset name ('small', 'medium', 'large').
        min_samples_per_genre: Minimum track count per genre used.
        data_dir: Path to the cached data directory.
        processors_dir: Path to the cached preprocessor directory.
    """

    def __init__(
        self,
        subset: Optional[str] = None,
        min_samples_per_genre: Optional[int] = None,
        dataset_id: Optional[str] = None,
    ):
        """
        Initialize the data loader for a specific dataset configuration.

        The dataset can be specified either by explicit subset/min_samples
        or by a precomputed dataset_id. If dataset_id is provided, subset
        and min_samples are extracted from it (best effort).

        Args:
            subset: FMA subset name. If None, uses configured default.
            min_samples_per_genre: Minimum tracks per genre. If None, uses 100.
            dataset_id: Full dataset identifier string (e.g., 'medium_100').
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
                    self.min_samples_per_genre = 100
            else:
                self.subset = subset or paths.active_subset
                self.min_samples_per_genre = min_samples_per_genre or 100
        else:
            self.subset = subset or paths.active_subset
            self.min_samples_per_genre = min_samples_per_genre or 100
            self.dataset_id = f"{self.subset}_{self.min_samples_per_genre}"

        self.data_dir = paths.fma_features_dataset_dir / self.dataset_id
        self.processors_dir = paths.processors_data_dir / "fma"

        self._data: Optional[Dict[str, Any]] = None

        logger.debug(
            "LoadProcessedData initialized — dataset_id=%s, data_dir=%s",
            self.dataset_id,
            self.data_dir,
        )

    def _get_cache_file(self) -> Path:
        """Path to the full pipeline pickle cache file."""
        return self.processors_dir / f"pipeline_{self.dataset_id}.pkl"

    def _get_metadata_path(self) -> Path:
        """Path to the dataset metadata JSON file."""
        return self.data_dir / "metadata.json"

    def exists(self) -> bool:
        """
        Check whether cached data exists for this dataset configuration.

        Returns:
            True if the cache directory contains all required files.
        """
        has_files = (
            self._get_cache_file().exists()
            and self._get_metadata_path().exists()
            and (self.data_dir / "X_train.npy").exists()
        )
        if not has_files:
            logger.debug("Cache not found for dataset '%s'", self.dataset_id)
        return has_files

    def list_available_datasets(self) -> List[str]:
        """
        Discover all cached dataset configurations on disk.

        Scans the FMA features dataset directory for subdirectories
        containing a valid metadata.json file.

        Returns:
            Sorted list of dataset_id strings.
        """
        if not paths.fma_features_dataset_dir.exists():
            logger.debug(
                "Dataset directory not found: %s",
                paths.fma_features_dataset_dir,
            )
            return []

        datasets = sorted([
            d.name
            for d in paths.fma_features_dataset_dir.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ])

        logger.debug("Found %d cached dataset(s): %s", len(datasets), datasets)
        return datasets

    def load(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load preprocessed data from the disk cache.

        Reads .npy arrays for features and labels, JSON metadata,
        joblib preprocessor state, and optional track indices.

        Args:
            force_reload: If True, bypasses the in-memory cache and
                          re-reads from disk.

        Returns:
            Data dictionary with keys:
            - X_train, X_val, X_test: normalized feature arrays
            - y_train, y_val, y_test: encoded label arrays
            - train_indices, val_indices, test_indices: track ID arrays
            - label_encoder: fitted LabelEncoder (if available)
            - scaler: fitted StandardScaler (if available)
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

        logger.info("Loading cached data [%s] from: %s", self.dataset_id, self.data_dir)

        X_train = np.load(self.data_dir / "X_train.npy")
        X_val = np.load(self.data_dir / "X_val.npy")
        X_test = np.load(self.data_dir / "X_test.npy")
        y_train = np.load(self.data_dir / "y_train.npy")
        y_val = np.load(self.data_dir / "y_val.npy")
        y_test = np.load(self.data_dir / "y_test.npy")

        train_indices_path = self.data_dir / "train_indices.npy"
        if train_indices_path.exists():
            train_indices = np.load(train_indices_path)
            val_indices = np.load(self.data_dir / "val_indices.npy")
            test_indices = np.load(self.data_dir / "test_indices.npy")
        else:
            train_indices = val_indices = test_indices = None
            logger.debug("Track indices not found in cache")

        with open(self._get_metadata_path(), "r") as f:
            metadata = json.load(f)

        label_encoder = None
        scaler = None
        preprocessor_path = self.processors_dir / f"preprocessor_{self.dataset_id}.pkl"
        if preprocessor_path.exists():
            import joblib
            preprocessor_state = joblib.load(preprocessor_path)
            label_encoder = preprocessor_state["label_encoder"]
            scaler = preprocessor_state["scaler"]
        else:
            logger.warning("Preprocessor file not found: %s", preprocessor_path)

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
            "label_encoder": label_encoder,
            "scaler": scaler,
            "genre_names": metadata.get("class_names", []),
            "metadata": metadata,
            "dataset_id": self.dataset_id,
        }

        logger.info(
            "Loaded dataset [%s]: %d train, %d val, %d test — %d genres, %d features",
            self.dataset_id,
            len(X_train),
            len(X_val),
            len(X_test),
            metadata.get("num_classes", 0),
            metadata.get("num_features", 0),
        )

        return self._data

    def load_to_dataframe(self) -> Dict[str, Any]:
        """
        Load data and reconstruct pandas DataFrames with column names.

        If feature names are available in metadata, the feature arrays
        are wrapped in DataFrames with the original column labels.
        Otherwise, raw numpy arrays are returned.

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

    def print_info(self) -> None:
        """
        Print a human-readable summary of the cached dataset.

        This is a manual debugging/exploration utility. Does not
        load the full data — only reads metadata.
        """
        if not self.exists():
            print(f"Dataset not found: '{self.dataset_id}'")
            available = self.list_available_datasets()
            print(f"Available datasets: {available if available else 'none'}")
            return

        with open(self._get_metadata_path(), "r") as f:
            meta = json.load(f)

        print("=" * 60)
        print("CACHED DATASET INFO")
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
        print(f"  Processors:   {self.processors_dir}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def load_data(
    subset: Optional[str] = None,
    min_samples_per_genre: int = 100,
    dataset_id: Optional[str] = None,
    as_dataframe: bool = False,
) -> Dict[str, Any]:
    """
    One-liner to load preprocessed FMA data from cache.

    Args:
        subset: FMA subset ('small', 'medium', 'large').
                Defaults to the configured active subset.
        min_samples_per_genre: Minimum tracks per genre (default: 100).
        dataset_id: Full dataset ID string. Overrides subset/min if provided.
        as_dataframe: If True, returns feature data as pandas DataFrames
                      with column names instead of raw numpy arrays.

    Returns:
        Data dictionary with X_train, y_train, etc.

    Example:
        data = load_data(subset="medium", min_samples_per_genre=100)
        X_train, y_train = data['X_train'], data['y_train']
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
    min_samples_per_genre: int = 100,
    dataset_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quickly load only the train/val/test track index arrays.

    Args:
        subset: FMA subset name.
        min_samples_per_genre: Minimum tracks per genre.
        dataset_id: Full dataset ID (overrides subset/min).

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
    List all cached dataset configurations available on disk.

    Returns:
        Sorted list of dataset_id strings.

    Example:
        for ds in list_datasets():
            print(ds)
    """
    loader = LoadProcessedData()
    return loader.list_available_datasets()


# Short aliases for interactive use
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