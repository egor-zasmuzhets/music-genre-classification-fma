"""
src/data/loader.py
FMA metadata loader — loads raw CSV files from the Free Music Archive dataset.

Provides lazy-loading access to tracks, features, and genres metadata
with subset filtering and official train/val/test split extraction.

Typical usage:
    from src.data.loader import FMALoader

    loader = FMALoader()

    # Load medium subset tracks
    tracks = loader.get_tracks_by_subset("medium")

    # Get precomputed features
    features = loader.features  # (106574, 518)

    # Get official split indices
    splits = loader.get_available_splits(tracks)
    train_idx = splits['training']
"""

import logging
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

from src.utils.config import paths


logger = logging.getLogger(__name__)


class FMALoader:
    """
    Lazy loader for FMA metadata CSV files.

    Loads and caches three core metadata files from the FMA dataset:
    - tracks.csv: multi-index DataFrame (2 header levels) with track metadata
    - features.csv: multi-index DataFrame (3 header levels) with 518 precomputed features
    - genres.csv: single-index DataFrame with genre hierarchy

    All files are loaded lazily on first access and cached for subsequent calls.

    Attributes:
        metadata_dir: Path to the directory containing FMA metadata CSV files.
    """

    def __init__(self, metadata_dir: Optional[Path] = None):
        """
        Initialize the FMA loader.

        Args:
            metadata_dir: Path to the directory containing FMA metadata CSV files.
                          Defaults to the path configured in configs/paths.yaml.
        """
        self.metadata_dir = metadata_dir or paths.metadata_dir
        self._tracks: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._genres: Optional[pd.DataFrame] = None

        logger.debug("FMALoader initialized (lazy — no data loaded yet)")

    @property
    def tracks(self) -> pd.DataFrame:
        """
        Load tracks.csv with multi-index columns.

        The tracks file uses a two-level column header:
        - Level 0: broad category ('track', 'artist', 'album', 'set')
        - Level 1: specific field ('genre_top', 'title', 'subset', 'split', etc.)

        Returns:
            DataFrame with track metadata, indexed by track ID.

        Raises:
            FileNotFoundError: If tracks.csv is not found in metadata_dir.
        """
        if self._tracks is None:
            tracks_path = self.metadata_dir / "tracks.csv"
            if not tracks_path.exists():
                raise FileNotFoundError(f"tracks.csv not found: {tracks_path}")

            self._tracks = pd.read_csv(
                tracks_path,
                header=[0, 1],
                index_col=0,
                low_memory=False
            )
            logger.info(
                "Loaded tracks: %d rows × %d columns",
                self._tracks.shape[0],
                self._tracks.shape[1]
            )

        return self._tracks

    @property
    def features(self) -> pd.DataFrame:
        """
        Load features.csv with 518 precomputed features.

        The features file uses a three-level column header:
        - Level 0: 'feature'
        - Level 1: analysis method (e.g., 'mfcc', 'spectral', 'tonnetz')
        - Level 2: statistic (e.g., 'mean', 'std', 'skew', 'kurtosis')

        Returns:
            DataFrame with precomputed features, indexed by track ID.

        Raises:
            FileNotFoundError: If features.csv is not found in metadata_dir.
        """
        if self._features is None:
            features_path = self.metadata_dir / "features.csv"
            if not features_path.exists():
                raise FileNotFoundError(f"features.csv not found: {features_path}")

            self._features = pd.read_csv(
                features_path,
                header=[0, 1, 2],
                index_col=0,
                low_memory=False
            )
            logger.info(
                "Loaded features: %d rows × %d columns",
                self._features.shape[0],
                self._features.shape[1]
            )

        return self._features

    @property
    def genres(self) -> pd.DataFrame:
        """
        Load genres.csv with genre hierarchy.

        Contains genre metadata including parent-child relationships
        and genre titles.

        Returns:
            DataFrame with genre information, indexed by genre ID.

        Raises:
            FileNotFoundError: If genres.csv is not found in metadata_dir.
        """
        if self._genres is None:
            genres_path = self.metadata_dir / "genres.csv"
            if not genres_path.exists():
                raise FileNotFoundError(f"genres.csv not found: {genres_path}")

            self._genres = pd.read_csv(
                genres_path,
                index_col=0
            )
            logger.info(
                "Loaded genres: %d rows",
                self._genres.shape[0]
            )

        return self._genres

    def get_tracks_by_subset(self, subset: str = "medium") -> pd.DataFrame:
        """
        Filter tracks by FMA subset.

        The FMA dataset is divided into subsets based on track count:
        - small: 8,000 tracks
        - medium: 25,000 tracks (small + 17,000 more)
        - large: 106,574 tracks (the complete dataset)

        Medium subset is defined as all tracks NOT in the 'large' subset,
        which includes both small and medium collections.

        Args:
            subset: One of 'small', 'medium', or 'large'.

        Returns:
            DataFrame containing only tracks from the specified subset.

        Raises:
            KeyError: If the subset column is not found in the tracks data.
        """
        tracks = self.tracks

        subset_column = ('set', 'subset')

        if subset_column in tracks.columns:
            if subset == 'small':
                filtered = tracks[tracks[subset_column] == 'small'].copy()
            elif subset == 'medium':
                filtered = tracks[tracks[subset_column] != 'large'].copy()
            elif subset == 'large':
                filtered = tracks.copy()
            else:
                raise ValueError(
                    f"Unknown subset: '{subset}'. "
                    f"Expected: 'small', 'medium', or 'large'."
                )
            logger.info(
                "Filtered %s subset: %d tracks (from %d total)",
                subset,
                len(filtered),
                len(tracks)
            )
        else:
            raise KeyError(
                f"Subset column {subset_column} not found in tracks. "
                f"Available columns: {list(tracks.columns)}"
            )

        return filtered

    def get_available_splits(self, tracks_df: pd.DataFrame) -> Dict[str, pd.Index]:
        """
        Extract official train/validation/test split indices.

        The FMA dataset provides a predefined split in the ('set', 'split')
        column with values 'training', 'validation', and 'test'.

        Args:
            tracks_df: DataFrame filtered by subset, must contain
                       the ('set', 'split') column.

        Returns:
            Dictionary mapping split names to track index arrays:
            {'training': Index, 'validation': Index, 'test': Index}

        Raises:
            KeyError: If the split column is not found in the DataFrame.
        """
        split_column = ('set', 'split')

        if split_column not in tracks_df.columns:
            raise KeyError(
                f"Split column {split_column} not found. "
                f"Available columns: {list(tracks_df.columns)}"
            )

        splits = {
            'training': tracks_df[tracks_df[split_column] == 'training'].index,
            'validation': tracks_df[tracks_df[split_column] == 'validation'].index,
            'test': tracks_df[tracks_df[split_column] == 'test'].index
        }

        total = sum(len(v) for v in splits.values())
        if total != len(tracks_df):
            unlabeled_count = len(tracks_df) - total
            logger.warning(
                "%d tracks (%.1f%%) have no split label — they will be excluded",
                unlabeled_count,
                100 * unlabeled_count / len(tracks_df)
            )

        logger.debug(
            "Split sizes — train: %d, val: %d, test: %d",
            len(splits['training']),
            len(splits['validation']),
            len(splits['test'])
        )

        return splits

    def get_genre_mapping(self) -> Dict[int, str]:
        """
        Build a mapping from genre ID to genre name.

        Returns:
            Dictionary of the form {genre_id: genre_name}.
        """
        mapping = self.genres['title'].to_dict()
        logger.debug("Genre mapping built: %d genres", len(mapping))
        return mapping

    def print_info(self) -> None:
        """
        Print a human-readable summary of loaded metadata.

        This is a manual debugging/exploration utility.
        Triggers lazy loading of all data if not already cached.
        """
        print("=" * 50)
        print("FMALoader Info")
        print("=" * 50)
        print(f"Metadata directory: {self.metadata_dir}")

        if self._tracks is not None:
            print(f"Tracks: {self._tracks.shape}")
        else:
            print("Tracks: not loaded (lazy)")

        if self._features is not None:
            print(f"Features: {self._features.shape}")
        else:
            print("Features: not loaded (lazy)")

        if self._genres is not None:
            print(f"Genres: {self._genres.shape}")
        else:
            print("Genres: not loaded (lazy)")

        if self._tracks is not None:
            try:
                subsets = self.tracks[('set', 'subset')].unique()
                print(f"Subsets available: {subsets}")
            except KeyError:
                print("Subsets: column not found")

            try:
                available_splits = self.tracks[('set', 'split')].unique()
                print(f"Splits available: {available_splits}")
            except KeyError:
                print("Splits: column not found")