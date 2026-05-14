"""
src/data/preprocessor.py
Data preprocessing — label encoding, feature normalization, rare genre filtering.

Provides a scikit-learn-compatible preprocessing pipeline that:
- Filters out genres with insufficient samples
- Encodes string genre labels to integer class IDs
- Normalizes feature vectors using StandardScaler
- Computes balanced class weights for imbalanced datasets

All fit operations are performed exclusively on the training set to prevent
data leakage. Validation and test sets are only transformed.

Typical usage:
    from src.data.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor(min_samples_per_genre=100)

    # Filter rare genres
    tracks_filtered = preprocessor.filter_rare_genres(tracks, ('track', 'genre_top'))

    # Encode labels (fit on train only)
    y_train, y_val, y_test = preprocessor.encode_labels(y_train_raw, y_val_raw, y_test_raw)

    # Normalize features (fit on train only)
    X_train, X_val, X_test = preprocessor.normalize_features(X_train_raw, X_val_raw, X_test_raw)

    # Get class weights for loss weighting
    class_weights = preprocessor.get_class_weights(y_train)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessing pipeline for genre classification.

    Handles the full sequence of data preparation:
    1. Filtering rare genres based on minimum sample threshold
    2. Label encoding (string genre names → integer class IDs)
    3. Feature normalization (zero mean, unit variance)
    4. Class weight computation for imbalanced loss functions

    The fit/transform split follows scikit-learn conventions:
    fit is always called on training data only, transform is applied
    to validation and test sets to prevent information leakage.

    Attributes:
        min_samples_per_genre: Minimum track count required to retain a genre.
        label_encoder: Fitted sklearn LabelEncoder instance.
        scaler: Fitted sklearn StandardScaler instance.
    """

    def __init__(self, min_samples_per_genre: int = 100):
        """
        Initialize the preprocessor.

        Args:
            min_samples_per_genre: Minimum number of tracks a genre must have
                                   to be retained. Genres with fewer samples
                                   are discarded. Default is 100.
        """
        self.min_samples_per_genre = min_samples_per_genre
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._is_fitted = False

        logger.debug(
            "DataPreprocessor initialized (min_samples_per_genre=%d)",
            min_samples_per_genre
        )

    def filter_rare_genres(
        self,
        tracks_df: pd.DataFrame,
        genre_col: Tuple[str, str]
    ) -> pd.DataFrame:
        """
        Remove genres with fewer than min_samples_per_genre tracks.

        Analyzes the distribution of genres in the dataset and retains
        only those meeting the minimum sample threshold. Tracks belonging
        to rare genres are dropped entirely.

        Args:
            tracks_df: DataFrame containing track metadata with genre labels.
            genre_col: Multi-index column tuple identifying the genre column
                       (e.g., ('track', 'genre_top')).

        Returns:
            Filtered DataFrame containing only tracks from sufficiently
            represented genres.

        Raises:
            KeyError: If the specified genre column is not found in the DataFrame.
        """
        if genre_col not in tracks_df.columns:
            raise KeyError(
                f"Genre column {genre_col} not found. "
                f"Available columns: {list(tracks_df.columns)}"
            )

        genre_counts = tracks_df[genre_col].value_counts()
        rare_mask = genre_counts < self.min_samples_per_genre
        rare_genres = genre_counts[rare_mask].index.tolist()
        common_genres = genre_counts[~rare_mask].index.tolist()

        n_total_before = len(tracks_df)
        filtered = tracks_df[tracks_df[genre_col].isin(common_genres)].copy()
        n_total_after = len(filtered)
        n_removed = n_total_before - n_total_after

        if rare_genres:
            removed_tracks = genre_counts[rare_mask].sum()
            logger.info(
                "Removed %d rare genres (%d tracks): %s",
                len(rare_genres),
                removed_tracks,
                ", ".join(f"{g}({genre_counts[g]})" for g in rare_genres)
            )

        logger.info(
            "Genre filtering complete: %d genres retained, "
            "%d tracks remaining (removed %d, %.1f%%)",
            len(common_genres),
            n_total_after,
            n_removed,
            100 * n_removed / n_total_before if n_total_before else 0
        )

        return filtered

    def encode_labels(
        self,
        y_train: pd.Series,
        y_val: Optional[pd.Series] = None,
        y_test: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Encode string genre labels to integer class IDs.

        Fits the LabelEncoder on the training set and transforms validation
        and test sets using the same encoding. This ensures consistent
        label-to-ID mapping across all splits.

        Args:
            y_train: Training set genre labels (raw strings).
            y_val: Validation set genre labels, or None.
            y_test: Test set genre labels, or None.

        Returns:
            Tuple of (y_train_encoded, y_val_encoded, y_test_encoded).
            val/test entries are None if the corresponding input was None.
        """
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_classes = len(self.label_encoder.classes_)

        y_val_encoded = None
        if y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)

        y_test_encoded = None
        if y_test is not None:
            y_test_encoded = self.label_encoder.transform(y_test)

        logger.info("Label encoding complete: %d classes", n_classes)
        logger.debug(
            "Class mapping: %s",
            {i: name for i, name in enumerate(self.label_encoder.classes_)}
        )

        return y_train_encoded, y_val_encoded, y_test_encoded

    def normalize_features(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Normalize feature vectors to zero mean and unit variance.

        Fits the StandardScaler on the training set only and transforms
        validation and test sets using the learned statistics.

        Args:
            X_train: Training set feature matrix.
            X_val: Validation set feature matrix, or None.
            X_test: Test set feature matrix, or None.

        Returns:
            Tuple of (X_train_scaled, X_val_scaled, X_test_scaled).
            val/test entries are None if the corresponding input was None.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)

        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)

        self._is_fitted = True
        n_features = X_train_scaled.shape[1]

        logger.info(
            "Feature normalization complete: %d features, "
            "train mean=%.4f, train std=%.4f",
            n_features,
            float(X_train_scaled.mean()),
            float(X_train_scaled.std())
        )

        if X_val_scaled is not None:
            logger.debug(
                "Validation stats — mean=%.4f, std=%.4f",
                float(X_val_scaled.mean()),
                float(X_val_scaled.std())
            )
        if X_test_scaled is not None:
            logger.debug(
                "Test stats — mean=%.4f, std=%.4f",
                float(X_test_scaled.mean()),
                float(X_test_scaled.std())
            )

        return X_train_scaled, X_val_scaled, X_test_scaled

    def get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Compute balanced class weights for imbalanced datasets.

        Uses sklearn's 'balanced' strategy:
        weight = n_samples / (n_classes * n_samples_per_class)

        These weights can be passed directly to loss functions in
        scikit-learn, XGBoost, or PyTorch to penalize errors on
        underrepresented classes more heavily.

        Args:
            y_train: Encoded training labels (integer class IDs).

        Returns:
            Dictionary mapping class ID to its computed weight.
        """
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, weights))

        logger.info(
            "Class weights computed for %d classes (min=%.3f, max=%.3f, ratio=%.1f:1)",
            len(weight_dict),
            min(weight_dict.values()),
            max(weight_dict.values()),
            max(weight_dict.values()) / min(weight_dict.values())
            if min(weight_dict.values()) > 0 else float('inf')
        )
        logger.debug("Class weight details: %s", weight_dict)

        return weight_dict

    def save(self, path: Path) -> None:
        """
        Persist the preprocessor state to disk.

        Saves the fitted LabelEncoder, StandardScaler, and configuration
        parameters as a joblib archive. The saved preprocessor can be
        reloaded with load() for inference without refitting.

        Args:
            path: File path for the saved preprocessor (e.g., .pkl or .joblib).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'min_samples_per_genre': self.min_samples_per_genre
        }
        joblib.dump(state, path)
        logger.info("Preprocessor saved to: %s", path)

    def load(self, path: Path) -> None:
        """
        Restore a previously saved preprocessor state.

        Loads the LabelEncoder, StandardScaler, and configuration from
        a joblib archive created by save().

        Args:
            path: File path to the saved preprocessor archive.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")

        data = joblib.load(path)
        self.label_encoder = data['label_encoder']
        self.scaler = data['scaler']
        self.min_samples_per_genre = data['min_samples_per_genre']
        self._is_fitted = True

        logger.info(
            "Preprocessor loaded from: %s (%d classes, %s)",
            path,
            len(self.label_encoder.classes_),
            "fitted" if self._is_fitted else "not fitted"
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the preprocessor has been fitted on training data."""
        return self._is_fitted

    @property
    def class_names(self) -> List[str]:
        """
        List of genre class names in encoding order.

        Returns:
            List of genre name strings, where index corresponds to encoded label.

        Raises:
            RuntimeError: If the label encoder has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "LabelEncoder has not been fitted yet. "
                "Call encode_labels() first."
            )
        return list(self.label_encoder.classes_)

    @property
    def n_classes(self) -> int:
        """
        Number of unique classes after label encoding.

        Returns:
            Integer count of classes.

        Raises:
            RuntimeError: If the label encoder has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "LabelEncoder has not been fitted yet. "
                "Call encode_labels() first."
            )
        return len(self.label_encoder.classes_)

    @property
    def n_features(self) -> int:
        """
        Number of features the scaler was fitted on.

        Returns:
            Integer count of features.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "StandardScaler has not been fitted yet. "
                "Call normalize_features() first."
            )
        return self.scaler.n_features_in_

    def print_info(self) -> None:
        """
        Print a human-readable summary of the preprocessor state.

        This is a manual debugging/exploration utility. Shows
        configuration, fit status, and learned statistics if available.
        """
        print("=" * 50)
        print("DataPreprocessor Info")
        print("=" * 50)
        print(f"min_samples_per_genre: {self.min_samples_per_genre}")
        print(f"Fitted: {self._is_fitted}")

        if self._is_fitted:
            print(f"Classes: {self.n_classes}")
            print(f"Features: {self.n_features}")
            print("\nClass mapping:")
            for i, name in enumerate(self.label_encoder.classes_):
                print(f"  {i}: {name}")
        else:
            print("Not yet fitted — call encode_labels() and normalize_features()")