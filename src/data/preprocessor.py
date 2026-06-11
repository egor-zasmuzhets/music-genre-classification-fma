"""
Data preprocessing — label encoding, feature normalization, rare genre filtering.

Provides a scikit-learn-compatible preprocessing pipeline that:
- Filters out genres with insufficient samples
- Encodes string genre labels to integer class IDs
- Normalizes feature vectors using StandardScaler
- Computes balanced class weights for imbalanced datasets

All fit operations are performed exclusively on the training set to prevent
data leakage. Validation and test sets are only transformed.

V2 Changes:
- State is stored as JSON (safe, readable, cross-platform)
- Save/load use directory structure with multiple JSON files
- No legacy joblib/pickle support

Typical usage:
    from src.data.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor(min_samples_per_genre=100)

    # Filter rare genres
    tracks_filtered = preprocessor.filter_rare_genres(tracks, ('track', 'genre_top'))

    # Encode labels (fit on train only)
    y_train, y_val, y_test = preprocessor.encode_labels(y_train_raw, y_val_raw, y_test_raw)

    # Normalize features (fit on train only)
    X_train, X_val, X_test = preprocessor.normalize_features(X_train_raw, X_val_raw, X_test_raw)

    # Save/load (V2)
    preprocessor.save(Path("preprocessor/"))
    preprocessor.load(Path("preprocessor/"))
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

    V2: State is stored as JSON (safe, readable) in a directory.

    Attributes:
        min_samples_per_genre: Minimum track count required to retain a genre.
        label_encoder: Fitted sklearn LabelEncoder instance.
        scaler: Fitted sklearn StandardScaler instance.
    """

    def __init__(self, min_samples_per_genre: int = 100) -> None:
        """
        Initialize the preprocessor.

        Args:
            min_samples_per_genre: Minimum number of tracks a genre must have
                                   to be retained. Default is 100.
        """
        self.min_samples_per_genre = min_samples_per_genre
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._is_fitted = False

        logger.debug("DataPreprocessor initialized (min_samples_per_genre=%d)", min_samples_per_genre)

    def filter_rare_genres(
        self,
        tracks_df: pd.DataFrame,
        genre_col: Tuple[str, str]
    ) -> pd.DataFrame:
        """
        Remove genres with fewer than min_samples_per_genre tracks.

        Args:
            tracks_df: DataFrame containing track metadata with genre labels.
            genre_col: Multi-index column tuple (e.g., ('track', 'genre_top')).

        Returns:
            Filtered DataFrame with only sufficiently represented genres.

        Raises:
            KeyError: If the specified genre column is not found.
        """
        if genre_col not in tracks_df.columns:
            raise KeyError(f"Genre column {genre_col} not found")

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
                len(rare_genres), removed_tracks,
                ", ".join(f"{g}({genre_counts[g]})" for g in rare_genres[:5])
            )

        logger.info(
            "Genre filtering: %d genres retained, %d tracks (removed %d, %.1f%%)",
            len(common_genres), n_total_after, n_removed,
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

        Fits LabelEncoder on training set, transforms validation and test sets.

        Args:
            y_train: Training set genre labels (raw strings).
            y_val: Validation set genre labels, or None.
            y_test: Test set genre labels, or None.

        Returns:
            Tuple of (y_train_encoded, y_val_encoded, y_test_encoded).
        """
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_classes = len(self.label_encoder.classes_)

        y_val_encoded = self.label_encoder.transform(y_val) if y_val is not None else None
        y_test_encoded = self.label_encoder.transform(y_test) if y_test is not None else None

        logger.info("Label encoding complete: %d classes", n_classes)

        return y_train_encoded, y_val_encoded, y_test_encoded

    def normalize_features(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Normalize feature vectors to zero mean and unit variance.

        Fits StandardScaler on training set, transforms validation and test sets.

        Args:
            X_train: Training set feature matrix.
            X_val: Validation set feature matrix, or None.
            X_test: Test set feature matrix, or None.

        Returns:
            Tuple of (X_train_scaled, X_val_scaled, X_test_scaled).
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None

        self._is_fitted = True
        n_features = X_train_scaled.shape[1]

        logger.info(
            "Feature normalization: %d features, mean=%.4f, std=%.4f",
            n_features, float(X_train_scaled.mean()), float(X_train_scaled.std())
        )

        return X_train_scaled, X_val_scaled, X_test_scaled

    def get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Compute balanced class weights for imbalanced datasets.

        Uses sklearn's 'balanced' strategy:
        weight = n_samples / (n_classes * n_samples_per_class)

        Args:
            y_train: Encoded training labels (integer class IDs).

        Returns:
            Dictionary mapping class ID to its computed weight.
        """
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, weights))

        min_w = min(weight_dict.values()) if weight_dict else 0.0
        max_w = max(weight_dict.values()) if weight_dict else 0.0

        logger.info(
            "Class weights: %d classes (min=%.3f, max=%.3f, ratio=%.1f:1)",
            len(weight_dict), min_w, max_w, max_w / min_w if min_w > 0 else float('inf')
        )

        return weight_dict

    def _extract_label_encoder_state(self) -> Dict:
        """Extract LabelEncoder state as JSON-serializable dict."""
        return {
            "classes": self.label_encoder.classes_.tolist(),
            "type": "LabelEncoder",
            "version": 2
        }

    def _extract_scaler_state(self) -> Dict:
        """Extract StandardScaler state as JSON-serializable dict."""
        state = {
            "type": "StandardScaler",
            "version": 2,
            "n_features": getattr(self.scaler, 'n_features_in_', None)
        }
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            state["mean"] = self.scaler.mean_.tolist()
        if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
            state["scale"] = self.scaler.scale_.tolist()
        return state

    def _restore_label_encoder(self, state: Dict) -> None:
        """Restore LabelEncoder from JSON state."""
        classes = state.get("classes", [])
        if classes:
            self.label_encoder.classes_ = np.array(classes)

    def _restore_scaler(self, state: Dict) -> None:
        """Restore StandardScaler from JSON state."""
        mean = state.get("mean")
        scale = state.get("scale")

        if mean is not None:
            self.scaler.mean_ = np.array(mean)
        if scale is not None:
            self.scaler.scale_ = np.array(scale)

        n_features = state.get("n_features")
        if n_features is not None:
            self.scaler.n_features_in_ = n_features

    def save(self, path: Union[Path, str]) -> None:
        """
        Persist the preprocessor state to disk as JSON files.

        Creates a directory with three JSON files:
        - scaler.json: mean, scale, n_features
        - label_encoder.json: classes list
        - config.json: min_samples_per_genre, is_fitted

        Args:
            path: Directory path where JSON files will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        scaler_state = self._extract_scaler_state()
        with open(path / "scaler.json", 'w', encoding='utf-8') as f:
            json.dump(scaler_state, f, indent=2, ensure_ascii=False)

        le_state = self._extract_label_encoder_state()
        with open(path / "label_encoder.json", 'w', encoding='utf-8') as f:
            json.dump(le_state, f, indent=2, ensure_ascii=False)

        config = {
            "min_samples_per_genre": self.min_samples_per_genre,
            "is_fitted": self._is_fitted,
            "version": 2
        }
        with open(path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info("Preprocessor saved to: %s", path)

    def load(self, path: Union[Path, str]) -> None:
        """
        Restore a previously saved preprocessor state from JSON files.

        Expects a directory containing:
        - scaler.json
        - label_encoder.json
        - config.json

        Args:
            path: Directory path containing the JSON files.

        Raises:
            FileNotFoundError: If the directory does not exist.
            FileNotFoundError: If required JSON files are missing.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Preprocessor directory not found: {path}")

        # Load scaler (optional, may not exist for pre-fitted state)
        scaler_path = path / "scaler.json"
        if scaler_path.exists():
            with open(scaler_path, 'r', encoding='utf-8') as f:
                scaler_state = json.load(f)
            self._restore_scaler(scaler_state)

        # Load label encoder (optional)
        le_path = path / "label_encoder.json"
        if le_path.exists():
            with open(le_path, 'r', encoding='utf-8') as f:
                le_state = json.load(f)
            self._restore_label_encoder(le_state)

        # Load config
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.min_samples_per_genre = config.get("min_samples_per_genre", 100)
        self._is_fitted = config.get("is_fitted", True)

        n_classes = len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 0
        logger.info("Preprocessor loaded from: %s (%d classes, fitted=%s)", path, n_classes, self._is_fitted)

    @property
    def is_fitted(self) -> bool:
        """Whether the preprocessor has been fitted on training data."""
        return self._is_fitted

    @property
    def class_names(self) -> List[str]:
        """
        List of genre class names in encoding order.

        Returns:
            List of genre name strings, index corresponds to encoded label.

        Raises:
            RuntimeError: If the label encoder has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder has not been fitted yet. Call encode_labels() first.")
        return list(self.label_encoder.classes_)

    @property
    def n_classes(self) -> int:
        """
        Number of unique classes after label encoding.

        Raises:
            RuntimeError: If the label encoder has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder has not been fitted yet. Call encode_labels() first.")
        return len(self.label_encoder.classes_)

    @property
    def n_features(self) -> int:
        """
        Number of features the scaler was fitted on.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("StandardScaler has not been fitted yet. Call normalize_features() first.")
        return self.scaler.n_features_in_

    def print_info(self) -> None:
        """Print a human-readable summary of the preprocessor state."""
        print("=" * 50)
        print("DataPreprocessor Info")
        print("=" * 50)
        print(f"min_samples_per_genre: {self.min_samples_per_genre}")
        print(f"Fitted: {self._is_fitted}")

        if self._is_fitted:
            print(f"Classes: {self.n_classes}")
            print(f"Features: {self.n_features}")
            print("\nClass mapping:")
            for i, name in enumerate(self.label_encoder.classes_[:10]):
                print(f"  {i}: {name}")
            if self.n_classes > 10:
                print(f"  ... and {self.n_classes - 10} more")
        else:
            print("Not yet fitted — call encode_labels() and normalize_features()")