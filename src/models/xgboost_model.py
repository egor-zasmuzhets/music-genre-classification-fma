"""
XGBoost classifier for music genre classification with comprehensive evaluation.

Provides training with optional class weights, early stopping on validation,
model persistence with metadata, and a multi-metric evaluation suite including
Top-k accuracy, confidence analysis, and a composite score for model selection.

Stacking support (v2):
- Added get_oof_predictions() for Out-of-Fold predictions
- Added predict_proba_for_stacking() for consistent interface
- Added get_base_predictions() for ensemble compatibility

Typical usage:
    from src.models.xgboost_model import XGBoostGenreClassifier

    model = XGBoostGenreClassifier(use_class_weights=True)
    model.fit(X_train, y_train, X_val, y_val, genre_names=genre_names)

    metrics = model.comprehensive_evaluate(X_test, y_test)
    print(f"Composite score: {metrics['composite_score']:.4f}")

    model.save(name="experiment_v1")
    model.load(name="experiment_v1")

    # For stacking ensemble
    oof_predictions = model.get_oof_predictions(X_train, y_train, n_folds=5)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from src.utils.config import paths as project_paths
from src.data.load_processed import load_data


logger = logging.getLogger(__name__)


class XGBoostGenreClassifier:
    """
    XGBoost classifier for single-label music genre classification.

    Supports class-weighted training to handle genre imbalance,
    model persistence with metadata, and a comprehensive evaluation
    suite combining multiple metrics into a composite score.

    V2: Uses XGBoost native .json format for model and separate .meta.json.
    Stacking: Supports Out-of-Fold predictions for ensemble training.

    Attributes:
        params: XGBoost hyperparameters dictionary.
        use_class_weights: Whether to compute balanced sample weights.
        model_name: Identifier used for save/load paths.
        sklearn_model: The underlying XGBClassifier instance.
        class_weights: Per-class weight mapping.
        genre_names: Ordered list of genre name strings.
    """

    DEFAULT_MODEL_NAME = "xgboost.json"

    def __init__(
        self,
        config_path: Optional[Path] = None,
        params: Optional[Dict[str, Any]] = None,
        use_class_weights: bool = True,
        random_state: int = 42,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the XGBoost classifier.

        Args:
            config_path: Path to models.yaml config file.
            params: XGBoost parameters dict. Overrides config.
            use_class_weights: Compute balanced class weights during training.
            random_state: Random seed for reproducibility.
            model_name: Filename stem for save/load.
        """
        if config_path is None:
            config_path = project_paths.configs_dir / "models.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {"parameters": {}, "training": {}}

        self.params = params or self.config.get("parameters", {}).copy()
        self.params["random_state"] = random_state
        self.params.setdefault("objective", "multi:softmax")

        self.use_class_weights = use_class_weights
        self.model_name = model_name or self.DEFAULT_MODEL_NAME

        self.model: Optional[xgb.Booster] = None
        self.sklearn_model: Optional[xgb.XGBClassifier] = None
        self._is_fitted = False
        self.class_weights: Optional[Dict[int, float]] = None
        self.genre_names: Optional[List[str]] = None

        logger.debug(
            "XGBoostGenreClassifier initialized — model_name=%s, use_class_weights=%s",
            self.model_name, use_class_weights
        )

    @property
    def _default_save_dir(self) -> Path:
        """Directory where model files are saved by default."""
        return project_paths.xgboost.models_dir

    def _get_default_path(self, filename: Optional[str] = None) -> Path:
        """Resolve the default save path for a model file."""
        name = filename or self.model_name
        if not name.endswith(".json"):
            name = name + ".json"
        return self._default_save_dir / name

    def _get_sample_weights(self, y_train: np.ndarray) -> Optional[np.ndarray]:
        """Compute per-sample weights for imbalanced classes."""
        if not self.use_class_weights:
            return None

        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        self.class_weights = dict(zip(classes, weights))

        sample_weights = np.array([self.class_weights[y] for y in y_train])

        min_w = min(weights) if len(weights) > 0 else 0.0
        max_w = max(weights) if len(weights) > 0 else 0.0

        logger.info(
            "Class weights computed — min=%.3f, max=%.3f, ratio=%.1f:1",
            min_w, max_w, max_w / min_w if min_w > 0 else float("inf")
        )

        return sample_weights

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        genre_names: Optional[List[str]] = None,
    ) -> "XGBoostGenreClassifier":
        """
        Train the XGBoost classifier.

        Args:
            X_train: Training feature matrix.
            y_train: Encoded training labels.
            X_val: Validation feature matrix for early stopping.
            y_val: Validation labels for early stopping.
            genre_names: Ordered list of genre names.

        Returns:
            Self (fitted classifier).
        """
        self.genre_names = genre_names
        num_classes = len(np.unique(y_train))
        self.params["num_class"] = num_classes
        sample_weights = self._get_sample_weights(y_train)

        logger.info(
            "Training XGBoost — samples=%d, features=%d, classes=%d, "
            "early_stopping=%s, class_weights=%s",
            len(X_train), X_train.shape[1], num_classes,
            X_val is not None, self.use_class_weights
        )
        logger.debug("XGBoost params: %s", self.params)

        self.sklearn_model = xgb.XGBClassifier(
            **self.params,
        )

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        self.sklearn_model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False,
        )

        self.model = self.sklearn_model.get_booster()
        self._is_fitted = True

        best_iter = getattr(self.sklearn_model, "best_iteration", None)
        best_score = getattr(self.sklearn_model, "best_score", float("nan"))

        logger.info("Training complete — best_iteration=%s, best_score=%.4f", best_iter, best_score)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_fitted()
        return self.sklearn_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_fitted()
        return self.sklearn_model.predict_proba(X)

    def predict_proba_for_stacking(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for predict_proba for consistent interface with CNN in stacking ensemble.
        """
        return self.predict_proba(X)

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model is not fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

    def get_oof_predictions(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Generate Out-of-Fold (OOF) predictions for stacking ensemble.

        Uses stratified k-fold cross-validation to get unbiased predictions
        on training data for training the meta-model. This prevents overfitting
        that would occur if we used the same model's predictions on its own
        training data.

        Args:
            X_train: Training feature matrix (n_samples, n_features)
            y_train: Training labels (n_samples,)
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            verbose: Whether to log progress

        Returns:
            OOF predictions of shape (n_samples, n_classes) with probabilities
            from the fold where each sample was in the validation set.
        """
        from sklearn.model_selection import StratifiedKFold

        self._check_fitted()

        n_classes = self.params.get('num_class', len(np.unique(y_train)))
        oof_proba = np.zeros((len(X_train), n_classes))

        kf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )

        if verbose:
            logger.info(f"Generating OOF predictions with {n_folds} folds...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            # Create fresh model for this fold
            fold_model = XGBoostGenreClassifier(
                params=self.params.copy(),
                use_class_weights=self.use_class_weights,
                random_state=random_state + fold
            )

            # Train on fold
            fold_model.fit(
                X_train[train_idx], y_train[train_idx],
                X_val=None, y_val=None,
                genre_names=self.genre_names
            )

            # Predict on validation
            oof_proba[val_idx] = fold_model.predict_proba(X_train[val_idx])

            if verbose:
                logger.info(f"  Fold {fold + 1}/{n_folds} completed")

        if verbose:
            logger.info("OOF predictions generated successfully")

        return oof_proba

    def get_base_predictions(
        self,
        X: np.ndarray,
        return_type: str = "proba"
    ) -> np.ndarray:
        """
        Get base predictions for stacking ensemble (compatible with CNN interface).

        Args:
            X: Feature matrix
            return_type: One of 'proba', 'logits', or 'predict'

        Returns:
            Predictions of shape (n_samples, n_classes) for 'proba'/'logits',
            or (n_samples,) for 'predict'
        """
        self._check_fitted()

        if return_type == "proba":
            return self.predict_proba(X)
        elif return_type == "logits":
            # XGBoost doesn't have logits, return probabilities as approximation
            return self.predict_proba(X)
        elif return_type == "predict":
            return self.predict(X)
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

    def top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        k: int = 3,
    ) -> float:
        """
        Compute Top-k accuracy.

        A prediction is correct if the true label is among the top-k
        predicted classes by probability.
        """
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        return float(correct)

    def confidence_analysis(
        self,
        X_test: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze model confidence on correct vs. incorrect predictions.

        Returns:
            Dictionary with confidence statistics.
        """
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)

        max_proba = y_pred_proba.max(axis=1)
        is_correct = y_pred == y_true

        correct_conf = max_proba[is_correct]
        wrong_conf = max_proba[~is_correct]

        correct_mean = float(correct_conf.mean()) if len(correct_conf) > 0 else 0.0
        correct_std = float(correct_conf.std()) if len(correct_conf) > 0 else 0.0
        wrong_mean = float(wrong_conf.mean()) if len(wrong_conf) > 0 else 0.0
        wrong_std = float(wrong_conf.std()) if len(wrong_conf) > 0 else 0.0

        return {
            "confidence_correct_mean": correct_mean,
            "confidence_correct_std": correct_std,
            "confidence_wrong_mean": wrong_mean,
            "confidence_wrong_std": wrong_std,
            "confidence_gap": correct_mean - wrong_mean,
            "low_confidence_count": int(np.sum(max_proba < 0.5)),
            "low_confidence_rate": float(np.mean(max_proba < 0.5)),
        }

    @staticmethod
    def _compute_composite_score(metrics: Dict[str, Any]) -> float:
        """
        Compute weighted composite score for model comparison.

        Weights: F1-macro (40%), Top-3 accuracy (40%), F1-weighted (20%)
        """
        f1_macro = metrics.get("f1_macro", 0.0)
        top_3 = metrics.get("top_3_accuracy", 0.0)
        f1_weighted = metrics.get("f1_weighted", 0.0)

        return 0.4 * f1_macro + 0.4 * top_3 + 0.2 * f1_weighted

    def comprehensive_evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        genre_names: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run a comprehensive evaluation suite.

        Returns:
            Nested dictionary with all metrics.
        """
        if k_values is None:
            k_values = [1, 3, 5]

        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        target_names = genre_names or self.genre_names

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        }

        for k in k_values:
            metrics[f"top_{k}_accuracy"] = self.top_k_accuracy(y_test, y_pred_proba, k=k)

        metrics["confidence"] = self.confidence_analysis(X_test, y_test)

        if target_names:
            metrics["classification_report"] = classification_report(
                y_test, y_pred, target_names=target_names,
                output_dict=True, zero_division=0,
            )

        try:
            metrics["roc_auc_ovo"] = float(roc_auc_score(
                y_test, y_pred_proba, multi_class="ovo", average="weighted"
            ))
        except Exception:
            metrics["roc_auc_ovo"] = None

        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(
                y_test, y_pred_proba, multi_class="ovr", average="weighted"
            ))
        except Exception:
            metrics["roc_auc_ovr"] = None

        metrics["composite_score"] = self._compute_composite_score(metrics)

        logger.info(
            "Evaluation complete — accuracy=%.4f, f1_macro=%.4f, "
            "top_3=%.4f, composite=%.4f",
            metrics["accuracy"], metrics["f1_macro"],
            metrics.get("top_3_accuracy", 0.0), metrics["composite_score"]
        )

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Return the most important features according to the model.
        """
        self._check_fitted()

        importance = self.sklearn_model.feature_importances_

        try:
            data = load_data()
            feature_names = data["metadata"].get("feature_names")
            if feature_names is None or len(feature_names) != len(importance):
                feature_names = [f"feature_{i}" for i in range(len(importance))]
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        logger.info("Top-%d feature importance extracted", top_n)

        return df.head(top_n)

    def save(
        self,
        filepath: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> Path:
        """
        Save the trained model and metadata to disk.

        Model saved as XGBoost native .json format.
        Metadata saved as separate .meta.json file.
        """
        self._check_fitted()

        if filepath is not None:
            save_path = Path(filepath)
        elif name is not None:
            save_path = self._get_default_path(name)
        else:
            save_path = self._get_default_path()

        if save_path.suffix != ".json":
            save_path = save_path.with_suffix(".json")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.sklearn_model.save_model(str(save_path))

        meta_path = save_path.with_suffix(".meta.json")
        meta = {
            "params": self.params,
            "use_class_weights": self.use_class_weights,
            "genre_names": self.genre_names,
            "class_weights": (
                {int(k): float(v) for k, v in self.class_weights.items()}
                if self.class_weights else None
            ),
            "is_fitted": self._is_fitted,
            "task": "mono_classification",
            "description": "Single-label genre classification",
            "model_name": save_path.stem,
            "format_version": 2,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Model saved: %s", save_path)
        logger.info("Metadata saved: %s", meta_path)

        return save_path

    @classmethod
    def load(
            cls,
            filepath: Optional[Path] = None,
            name: Optional[str] = None,
    ) -> "XGBoostGenreClassifier":
        """
        Load a previously saved model and metadata.

        Expects XGBoost .json model file and companion .meta.json.
        """
        # Create temporary instance to use helper methods
        temp_instance = cls()

        if filepath is not None:
            load_path = Path(filepath)
        elif name is not None:
            load_path = temp_instance._get_default_path(name)
        else:
            load_path = temp_instance._get_default_path()

        if not load_path.exists():
            load_path = temp_instance._resolve_missing_path(load_path)

        sklearn_model = xgb.XGBClassifier()
        sklearn_model.load_model(str(load_path))
        model = sklearn_model.get_booster()

        meta_path = load_path.with_suffix(".meta.json")
        params = {}
        use_class_weights = True
        genre_names = None
        class_weights = None
        model_name = load_path.stem

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            params = meta.get("params", {})
            use_class_weights = meta.get("use_class_weights", True)
            genre_names = meta.get("genre_names")
            class_weights = meta.get("class_weights")
            model_name = meta.get("model_name", load_path.stem)

        # Create instance with loaded parameters
        instance = cls(
            params=params,
            use_class_weights=use_class_weights,
            model_name=model_name
        )
        instance.sklearn_model = sklearn_model
        instance.model = model
        instance.genre_names = genre_names
        instance.class_weights = class_weights
        instance._is_fitted = True

        logger.info("Model loaded: %s", load_path)

        return instance

    def _resolve_missing_path(self, load_path: Path) -> Path:
        """
        Attempt to find a model file when the exact path is missing.
        """
        alt_path = load_path.with_suffix("")
        if alt_path.exists():
            return alt_path

        models_dir = self._default_save_dir
        if models_dir.exists():
            candidates = list(models_dir.glob("*.json"))
            if candidates:
                fallback = candidates[0]
                logger.warning(
                    "Model not found at %s — loading most recent: %s",
                    load_path, fallback.name
                )
                return fallback

        available = [f.name for f in models_dir.glob("*.json")] if models_dir.exists() else []
        raise FileNotFoundError(
            f"Model not found.\nSearched: {load_path}\n"
            f"Directory: {models_dir}\nAvailable: {available if available else 'none'}"
        )

    def print_info(self) -> None:
        """Print a human-readable summary of the model state."""
        if not self._is_fitted:
            print("Model has not been fitted yet.")
            return

        print("=" * 60)
        print("XGBOOST CLASSIFIER SUMMARY (with Stacking Support)")
        print("=" * 60)
        print(f"Model name:      {self.model_name}")
        print(f"Classes:         {self.params.get('num_class', '?')}")
        print(f"Estimators:      {self.params.get('n_estimators', '?')}")
        print(f"Max depth:       {self.params.get('max_depth', '?')}")
        print(f"Learning rate:   {self.params.get('learning_rate', '?')}")
        print(f"Class weights:   {self.use_class_weights}")
        if self.class_weights:
            min_w = min(self.class_weights.values())
            max_w = max(self.class_weights.values())
            print(f"Weight range:    {min_w:.3f} – {max_w:.3f}")
        print("-" * 60)
        print("Stacking methods:")
        print("  - get_oof_predictions()      → Out-of-Fold predictions")
        print("  - predict_proba_for_stacking() → alias for predict_proba")
        print("  - get_base_predictions()     → unified interface")
        print("=" * 60)