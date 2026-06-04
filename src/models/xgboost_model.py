"""
XGBoost classifier for music genre classification with comprehensive evaluation.

Provides training with optional class weights, early stopping on validation,
model persistence with metadata, and a multi-metric evaluation suite including
Top-k accuracy, confidence analysis, and a composite score for model selection.

Typical usage:
    from src.models.xgboost_model import XGBoostGenreClassifier

    model = XGBoostGenreClassifier(use_class_weights=True)
    model.fit(X_train, y_train, X_val, y_val, genre_names=genre_names)

    metrics = model.comprehensive_evaluate(X_test, y_test)
    print(f"Composite score: {metrics['composite_score']:.4f}")

    model.save(name="experiment_v1")
    model.load(name="experiment_v1")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    Attributes:
        params: XGBoost hyperparameters dictionary.
        use_class_weights: Whether to compute balanced sample weights.
        model_name: Identifier used for save/load paths.
        sklearn_model: The underlying XGBClassifier instance (after fit/load).
        class_weights: Per-class weight mapping (if use_class_weights=True).
        genre_names: Ordered list of genre name strings.
    """

    DEFAULT_MODEL_NAME = "xgboost_auto.json"

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
            config_path: Path to a models.yaml config file.
                         Defaults to configs/models.yaml.
            params: XGBoost parameters dict. Overrides config file values.
            use_class_weights: Compute balanced class weights during training.
            random_state: Random seed for reproducibility.
            model_name: Filename stem for save/load. Defaults to 'xgboost_auto'.
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
            "XGBoostGenreClassifier initialized — model_name=%s, "
            "use_class_weights=%s",
            self.model_name,
            use_class_weights,
        )

    @property
    def _default_save_dir(self) -> Path:
        """Directory where model files are saved by default."""
        return project_paths.xgboost.models_dir

    def _get_default_path(self, filename: Optional[str] = None) -> Path:
        """
        Resolve the default save path for a model file.

        Args:
            filename: Base filename. Uses self.model_name if None.
                      '.json' extension is appended if missing.

        Returns:
            Absolute Path to the model JSON file.
        """
        name = filename or self.model_name
        if not name.endswith(".json"):
            name = name + ".json"
        return self._default_save_dir / name

    def _get_sample_weights(self, y_train: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute per-sample weights for imbalanced classes.

        Args:
            y_train: Encoded training labels.

        Returns:
            1D array of sample weights, or None if class weights are disabled.
        """
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
            min_w,
            max_w,
            max_w / min_w if min_w > 0 else float("inf"),
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
            X_train: Training feature matrix (n_samples, n_features).
            y_train: Encoded training labels (n_samples,).
            X_val: Validation feature matrix for early stopping.
            y_val: Validation labels for early stopping.
            genre_names: Ordered list of genre names. Stored for evaluation.

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
            len(X_train),
            X_train.shape[1],
            num_classes,
            X_val is not None,
            self.use_class_weights,
        )
        logger.debug("XGBoost params: %s", self.params)

        self.sklearn_model = xgb.XGBClassifier(
            **self.params,
            eval_metric="mlogloss",
        )

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        self.sklearn_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False,
        )

        self.model = self.sklearn_model.get_booster()
        self._is_fitted = True

        best_iter = getattr(self.sklearn_model, "best_iteration", None)
        best_score = getattr(self.sklearn_model, "best_score", float("nan"))

        logger.info(
            "Training complete — best_iteration=%s, best_score=%.4f",
            best_iter,
            best_score,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            1D array of integer class labels.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        return self.sklearn_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            2D array of shape (n_samples, n_classes).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        return self.sklearn_model.predict_proba(X)

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model is not fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

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

        Args:
            y_true: Ground truth labels.
            y_pred_proba: Predicted probability matrix.
            k: Number of top predictions to consider.

        Returns:
            Top-k accuracy as a float between 0 and 1.
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

        Computes mean and std of max predicted probability for correctly
        and incorrectly classified samples, plus the confidence gap.

        Args:
            X_test: Test feature matrix.
            y_true: Ground truth labels.

        Returns:
            Dictionary with keys: confidence_correct_mean,
            confidence_correct_std, confidence_wrong_mean,
            confidence_wrong_std, confidence_gap,
            low_confidence_count, low_confidence_rate.
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
        gap = correct_mean - wrong_mean if len(wrong_conf) > 0 else 0.0

        return {
            "confidence_correct_mean": correct_mean,
            "confidence_correct_std": correct_std,
            "confidence_wrong_mean": wrong_mean,
            "confidence_wrong_std": wrong_std,
            "confidence_gap": gap,
            "low_confidence_count": int(np.sum(max_proba < 0.5)),
            "low_confidence_rate": float(np.mean(max_proba < 0.5)),
        }

    @staticmethod
    def _compute_composite_score(metrics: Dict[str, Any]) -> float:
        """
        Compute a weighted composite score for model comparison.

        Weights:
            - F1-macro: 40% (emphasizes rare genre performance)
            - Top-3 accuracy: 40% (captures stylistic understanding)
            - F1-weighted: 20% (overall quality)

        Args:
            metrics: Dictionary with keys f1_macro, top_3_accuracy, f1_weighted.

        Returns:
            Composite score between 0 and 1.
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

        Computes accuracy, precision, recall, F1 (weighted/macro/micro),
        Top-k accuracy for each k, confidence analysis, ROC-AUC,
        and the composite score.

        Args:
            X_test: Test feature matrix.
            y_test: Ground truth labels.
            genre_names: Genre names for the classification report.
            k_values: List of k values for Top-k accuracy. Default: [1, 3, 5].

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
                y_test, y_pred_proba, multi_class="ovo", average="weighted",
            ))
        except Exception:
            metrics["roc_auc_ovo"] = None

        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(
                y_test, y_pred_proba, multi_class="ovr", average="weighted",
            ))
        except Exception:
            metrics["roc_auc_ovr"] = None

        metrics["composite_score"] = self._compute_composite_score(metrics)

        logger.info(
            "Evaluation complete — accuracy=%.4f, f1_macro=%.4f, "
            "top_3=%.4f, composite=%.4f",
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics.get("top_3_accuracy", 0.0),
            metrics["composite_score"],
        )

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Return the most important features according to the model.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with columns 'feature' and 'importance', sorted descending.

        Raises:
            RuntimeError: If the model has not been fitted.
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

        Args:
            filepath: Exact file path. Takes precedence over name.
            name: Base filename. Saved to the default model directory.

        Returns:
            Path to the saved model file.

        Raises:
            RuntimeError: If the model has not been fitted.
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
            "description": "Single-label genre classification (main genre only)",
            "model_name": save_path.stem,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Model saved: %s", save_path)
        logger.info("Metadata saved: %s", meta_path)

        return save_path

    def load(
        self,
        filepath: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> "XGBoostGenreClassifier":
        """
        Load a previously saved model and metadata.

        Args:
            filepath: Exact file path. Takes precedence over name.
            name: Base filename to load from the default model directory.

        Returns:
            Self (loaded classifier).

        Raises:
            FileNotFoundError: If no model file can be found.
        """
        if filepath is not None:
            load_path = Path(filepath)
        elif name is not None:
            load_path = self._get_default_path(name)
        else:
            load_path = self._get_default_path()

        if not load_path.exists():
            load_path = self._resolve_missing_path(load_path)

        self.sklearn_model = xgb.XGBClassifier()
        self.sklearn_model.load_model(str(load_path))
        self.model = self.sklearn_model.get_booster()

        meta_path = load_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.params = meta.get("params", {})
            self.use_class_weights = meta.get("use_class_weights", True)
            self.genre_names = meta.get("genre_names")
            self.class_weights = meta.get("class_weights")
            self.model_name = meta.get("model_name", load_path.stem)

        self._is_fitted = True

        logger.info("Model loaded: %s", load_path)

        return self

    def _resolve_missing_path(self, load_path: Path) -> Path:
        """
        Attempt to find a model file when the exact path is missing.

        Tries the path without extension, then falls back to the first
        .json file in the default model directory.

        Args:
            load_path: The path that was not found.

        Returns:
            A resolved Path that exists.

        Raises:
            FileNotFoundError: If no model file could be found.
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
                    load_path,
                    fallback.name,
                )
                return fallback

        available = (
            [f.name for f in models_dir.glob("*.json")]
            if models_dir.exists()
            else []
        )
        raise FileNotFoundError(
            f"Model not found.\n"
            f"Searched: {load_path}\n"
            f"Directory: {models_dir}\n"
            f"Available: {available if available else 'none'}"
        )

    def print_info(self) -> None:
        """
        Print a human-readable summary of the model state.

        This is a manual debugging/exploration utility.
        """
        if not self._is_fitted:
            print("Model has not been fitted yet.")
            return

        print("=" * 60)
        print("XGBOOST CLASSIFIER SUMMARY")
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
        print("=" * 60)