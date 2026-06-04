"""
Grid search for XGBoost hyperparameter tuning with comprehensive metrics.

Evaluates every parameter combination against multiple metrics including
F1-macro, Top-3 accuracy, F1-weighted, and a composite score that balances
rare-genre performance with overall quality. Results are persisted as CSV
and can be loaded for analysis.

Typical usage:
    from src.training.grid_search import XGBoostGridSearch, GRID_SMALL

    grid = XGBoostGridSearch(param_grid=GRID_SMALL, use_class_weights=True)
    grid.fit(X_train, y_train, X_val, y_val, X_test, y_test, genre_names)

    best_params = grid.get_best_params(metric="composite_score")
    grid.save_results()
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.models.xgboost_model import XGBoostGenreClassifier
from src.utils.config import paths


logger = logging.getLogger(__name__)


GRID_TEST = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100],
}

GRID_SMALL = {
    "max_depth": [3, 5, 7],
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.1, 0.3],
}

GRID_MEDIUM = {
    "max_depth": [3, 5, 7, 9],
    "n_estimators": [50, 100, 150, 200],
    "learning_rate": [0.1, 0.2, 0.3],
    "subsample": [0.8, 1.0],
    "min_child_weight": [1, 3],
}

GRID_FULL = {
    "max_depth": [3, 5, 7, 9, 11],
    "n_estimators": [50, 100, 150, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2, 0.3],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [0.5, 1, 1.5],
}


class XGBoostGridSearch:
    """
    Exhaustive hyperparameter search for XGBoost with multi-metric evaluation.

    Evaluates every parameter combination on:
    - composite_score (primary): 0.4×F1-macro + 0.4×Top-3 + 0.2×F1-weighted
    - f1_macro: emphasises rare genre performance
    - top_3_accuracy: captures stylistic understanding
    - accuracy: overall correctness

    Tracks best models by each metric and persists all results to CSV.

    Attributes:
        param_grid: Dictionary mapping parameter names to lists of values.
        use_class_weights: Whether to train with balanced class weights.
        results: List of per-combination result dicts (populated after fit).
        best_model_info: Summary of best models by each metric.
    """

    DEFAULT_RESULTS_NAME = "grid_search_results.csv"

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        use_class_weights: bool = True,
        results_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the grid search.

        Args:
            param_grid: Dictionary of parameter names to lists of candidate values.
            use_class_weights: Train each model with balanced class weights.
            results_name: Base filename for saving results (without extension).
                          Defaults to 'grid_search_results.csv'.
        """
        self.param_grid = param_grid
        self.use_class_weights = use_class_weights
        self.results_name = results_name or self.DEFAULT_RESULTS_NAME
        self.results: List[Dict[str, Any]] = []
        self.best_model_info: Optional[Dict[str, Any]] = None

        n_combinations = self._count_combinations()
        logger.info(
            "Grid search initialized — %d combinations, class_weights=%s",
            n_combinations,
            use_class_weights,
        )

    def _count_combinations(self) -> int:
        """Return the total number of hyperparameter combinations."""
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total

    @property
    def _default_save_dir(self) -> Path:
        """Directory where grid search results are saved by default."""
        return paths.xgboost.grid_search_dir

    def _get_default_path(self, filename: Optional[str] = None) -> Path:
        """
        Resolve the default save path for results.

        Args:
            filename: Base filename. Uses self.results_name if None.

        Returns:
            Absolute Path with .csv extension.
        """
        name = filename or self.results_name
        if not name.endswith(".csv"):
            name = name + ".csv"
        return self._default_save_dir / name

    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from the grid."""
        return list(ParameterGrid(self.param_grid))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        genre_names: Optional[List[str]] = None,
        save_intermediate: bool = True,
    ) -> pd.DataFrame:
        """
        Run the full grid search.

        Trains and evaluates an XGBoost model for every parameter
        combination. Intermediate results are saved every 5 combinations
        if save_intermediate is True.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data for early stopping.
            X_test, y_test: Test data for evaluation.
            genre_names: Ordered genre name strings.
            save_intermediate: Persist partial results every 5 combinations.

        Returns:
            DataFrame of all results sorted by composite_score descending.
        """
        param_combinations = self._get_param_combinations()
        total = len(param_combinations)

        logger.info(
            "Starting grid search — %d combinations, %d train / %d val / %d test",
            total, len(X_train), len(X_val), len(X_test),
        )

        best_composite = -1.0
        best_f1_macro = -1.0
        best_top3 = -1.0
        best_accuracy = -1.0

        best_composite_params: Optional[Dict[str, Any]] = None
        best_f1_macro_params: Optional[Dict[str, Any]] = None
        best_top3_params: Optional[Dict[str, Any]] = None
        best_accuracy_params: Optional[Dict[str, Any]] = None

        for i, params in enumerate(param_combinations, 1):
            start_time = time.time()

            model = XGBoostGenreClassifier(
                params=params,
                use_class_weights=self.use_class_weights,
            )

            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                genre_names=genre_names,
            )

            metrics = model.comprehensive_evaluate(X_test, y_test, genre_names)
            train_time = time.time() - start_time

            result = {
                **params,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "f1_micro": metrics["f1_micro"],
                "top_1_acc": metrics["top_1_accuracy"],
                "top_3_acc": metrics["top_3_accuracy"],
                "top_5_acc": metrics["top_5_accuracy"],
                "composite_score": metrics["composite_score"],
                "confidence_gap": metrics["confidence"]["confidence_gap"],
                "low_confidence_rate": metrics["confidence"]["low_confidence_rate"],
                "roc_auc_ovo": metrics.get("roc_auc_ovo", 0.0),
                "train_time": train_time,
            }
            self.results.append(result)

            if metrics["composite_score"] > best_composite:
                best_composite = metrics["composite_score"]
                best_composite_params = params.copy()

            if metrics["f1_macro"] > best_f1_macro:
                best_f1_macro = metrics["f1_macro"]
                best_f1_macro_params = params.copy()

            if metrics["top_3_accuracy"] > best_top3:
                best_top3 = metrics["top_3_accuracy"]
                best_top3_params = params.copy()

            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_accuracy_params = params.copy()

            logger.info(
                "[%d/%d] composite=%.4f, f1_macro=%.4f, top3=%.4f, time=%.1fs",
                i, total,
                metrics["composite_score"],
                metrics["f1_macro"],
                metrics["top_3_accuracy"],
                train_time,
            )

            if save_intermediate and i % 5 == 0:
                self.save_results()

        self.best_model_info = {
            "by_composite": {
                "params": best_composite_params,
                "score": best_composite,
            },
            "by_f1_macro": {
                "params": best_f1_macro_params,
                "score": best_f1_macro,
            },
            "by_top_3": {
                "params": best_top3_params,
                "score": best_top3,
            },
            "by_accuracy": {
                "params": best_accuracy_params,
                "score": best_accuracy,
            },
            "total_combinations": total,
            "completed_at": datetime.now().isoformat(),
        }

        self.save_results()

        logger.info(
            "Grid search complete — best composite=%.4f, "
            "best f1_macro=%.4f, best top3=%.4f, best accuracy=%.4f",
            best_composite,
            best_f1_macro,
            best_top3,
            best_accuracy,
        )

        return self.get_results()

    def get_results(self) -> pd.DataFrame:
        """
        Return all results as a DataFrame sorted by composite_score.

        Returns:
            DataFrame with one row per parameter combination.
        """
        df = pd.DataFrame(self.results)
        if len(df) > 0:
            return df.sort_values("composite_score", ascending=False)
        return df

    def get_best_params(self, metric: str = "composite_score") -> Dict[str, Any]:
        """
        Return the parameters of the best model by a given metric.

        Args:
            metric: Column name to maximize.
                    One of: 'composite_score', 'f1_macro', 'top_3_acc',
                    'accuracy', 'f1_weighted'.

        Returns:
            Dictionary of parameter names to their best values.

        Raises:
            ValueError: If no results are available.
        """
        results = self.get_results()
        if len(results) == 0:
            raise ValueError("No results available. Run fit() first.")

        best_row = results.loc[results[metric].idxmax()]
        params = {k: best_row[k] for k in self.param_grid.keys()}

        logger.info(
            "Best model by %s — composite=%.4f, f1_macro=%.4f, top3=%.4f, "
            "accuracy=%.4f",
            metric,
            best_row["composite_score"],
            best_row["f1_macro"],
            best_row["top_3_acc"],
            best_row["accuracy"],
        )

        return params

    def get_best_by_metric(self) -> Dict[str, Dict[str, Any]]:
        """
        Return best models by each tracked metric.

        Returns:
            Dictionary with keys 'by_composite', 'by_f1_macro',
            'by_top_3', 'by_accuracy'. Each value is a dict with keys
            'params' and 'metrics'.
        """
        results = self.get_results()
        if len(results) == 0:
            return {}

        def _best_row(column: str) -> Dict[str, Any]:
            row = results.loc[results[column].idxmax()]
            return {
                "params": {k: row[k] for k in self.param_grid.keys()},
                "metrics": row.to_dict(),
            }

        return {
            "by_composite": _best_row("composite_score"),
            "by_f1_macro": _best_row("f1_macro"),
            "by_top_3": _best_row("top_3_acc"),
            "by_accuracy": _best_row("accuracy"),
        }

    def save_results(
        self,
        filepath: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Save grid search results to CSV and metadata to JSON.

        Args:
            filepath: Exact file path. Takes precedence over name.
            name: Base filename. Saved to the default grid search directory.

        Returns:
            Path to the saved CSV file, or None if there are no results.
        """
        df = self.get_results()

        if len(df) == 0:
            logger.warning("No results to save — run fit() first")
            return None

        if filepath is not None:
            save_path = Path(filepath)
        elif name is not None:
            save_path = self._get_default_path(name)
        else:
            save_path = self._get_default_path()

        if save_path.suffix != ".csv":
            save_path = save_path.with_suffix(".csv")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Grid search results saved: %s", save_path)

        if self.best_model_info:
            meta_path = save_path.with_suffix(".meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.best_model_info, f, indent=2, ensure_ascii=False)
            logger.info("Grid search metadata saved: %s", meta_path)

        return save_path

    def load_results(
        self,
        filepath: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load previously saved grid search results from CSV.

        Args:
            filepath: Exact file path. Takes precedence over name.
            name: Base filename to load from the default directory.

        Returns:
            DataFrame with the loaded results.

        Raises:
            FileNotFoundError: If the results file cannot be found.
        """
        if filepath is not None:
            load_path = Path(filepath)
        elif name is not None:
            load_path = self._get_default_path(name)
        else:
            load_path = self._get_default_path()

        if not load_path.exists():
            alt_path = load_path.with_suffix("")
            if alt_path.exists():
                load_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Results file not found: {load_path}"
                )

        df = pd.read_csv(load_path)
        self.results = df.to_dict("records")

        logger.info(
            "Grid search results loaded: %s (%d combinations)",
            load_path,
            len(df),
        )

        return df

    def print_info(self) -> None:
        """
        Print a human-readable summary of grid search results.

        This is a manual debugging/exploration utility.
        """
        if len(self.results) == 0:
            print("No results yet. Run fit() first.")
            return

        results_df = self.get_results()

        print("=" * 70)
        print("GRID SEARCH SUMMARY")
        print("=" * 70)
        print(f"Total combinations: {len(self.results)}")
        print(
            f"Best composite_score: "
            f"{results_df.iloc[0]['composite_score']:.4f}"
        )
        print(f"Best F1-macro:       {results_df['f1_macro'].max():.4f}")
        print(f"Best Top-3 accuracy:  {results_df['top_3_acc'].max():.4f}")
        print(f"Best accuracy:        {results_df['accuracy'].max():.4f}")
        print("-" * 70)

        if self.best_model_info:
            print("\nBest parameters by metric:")
            for metric, info in self.best_model_info.items():
                if info["params"]:
                    print(f"  {metric}: score={info['score']:.4f}")
        print("=" * 70)