"""
src/training/train_xgboost.py
XGBoost training script with comprehensive evaluation (mono classification).

Orchestrates data loading, model training (or grid search), evaluation,
and result persistence. Supports command-line configuration for subset,
class weights, and hyperparameter search.

Usage:
    python -m src.training.train_xgboost --subset medium --min_samples 100
    python -m src.training.train_xgboost --grid_search --grid_size small
    python -m src.training.train_xgboost --subset small --no_weights
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.data.load_processed import load_data
from src.models.xgboost_model import XGBoostGenreClassifier
from src.training.analyzer import ModelAnalyzer
from src.utils.config import paths
from src.utils.logging_utils import setup_logging


logger = logging.getLogger(__name__)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for XGBoost training.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for mono classification"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to model config YAML file"
    )
    parser.add_argument(
        "--subset", type=str, default="medium",
        choices=["small", "medium", "large"],
        help="FMA subset to use (default: medium)"
    )
    parser.add_argument(
        "--min_samples", type=int, default=100,
        help="Minimum tracks per genre (default: 100)"
    )
    parser.add_argument(
        "--no_weights", action="store_true",
        help="Disable class-balanced sample weights"
    )
    parser.add_argument(
        "--grid_search", action="store_true",
        help="Run hyperparameter grid search before final training"
    )
    parser.add_argument(
        "--grid_size", type=str, default="small",
        choices=["small", "medium", "full"],
        help="Grid search coverage (default: small)"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    return parser


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_report_data(
    subset: str,
    min_samples: int,
) -> Dict[str, Any]:
    """
    Load preprocessed data and log a distribution summary.

    Args:
        subset: FMA subset name.
        min_samples: Minimum tracks per genre.

    Returns:
        Data dictionary from load_data().
    """
    logger.info("Loading data [subset=%s, min_samples=%d]", subset, min_samples)

    data = load_data(subset=subset, min_samples_per_genre=min_samples)

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    genre_names = data["genre_names"]

    logger.info(
        "Data loaded — train: %s, val: %s, test: %s, genres: %d",
        X_train.shape, X_val.shape, X_test.shape, len(genre_names)
    )

    # Log class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    distribution_lines = []
    for genre_id, count in zip(unique, counts):
        name = genre_names[genre_id] if genre_id < len(genre_names) else f"#{genre_id}"
        pct = 100 * count / len(y_train)
        distribution_lines.append(f"  {name:20s}: {count:5d} ({pct:5.1f}%)")

    logger.debug(
        "Training class distribution:\n%s",
        "\n".join(distribution_lines),
    )

    return data


# ============================================================================
# GRID SEARCH
# ============================================================================

def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    genre_names: list,
    grid_size: str,
    use_class_weights: bool,
) -> Optional[Dict[str, Any]]:
    """
    Run hyperparameter grid search and return best parameters.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data for final evaluation.
        genre_names: Ordered list of genre names.
        grid_size: 'small', 'medium', or 'full'.
        use_class_weights: Whether to use balanced class weights.

    Returns:
        Dictionary of best parameters, or None if grid search fails.
    """
    from src.training.grid_search import (
        XGBoostGridSearch,
        GRID_SMALL,
        GRID_MEDIUM,
        GRID_FULL,
    )

    grid_map = {
        "small": GRID_SMALL,
        "medium": GRID_MEDIUM,
        "full": GRID_FULL,
    }
    param_grid = grid_map[grid_size]

    logger.info(
        "Starting grid search [size=%s, combinations=%d]",
        grid_size,
        _count_combinations(param_grid),
    )

    grid_search = XGBoostGridSearch(
        param_grid=param_grid,
        use_class_weights=use_class_weights,
    )

    grid_search.fit(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        genre_names,
    )

    grid_search.save_results()

    best_by_metric = grid_search.get_best_by_metric()
    for metric, info in best_by_metric.items():
        logger.info(
            "Best by %s: composite=%.4f, f1_macro=%.4f, top3=%.4f",
            metric,
            info["metrics"]["composite_score"],
            info["metrics"]["f1_macro"],
            info["metrics"].get("top_3_accuracy", 0.0),
        )

    best_params = grid_search.get_best_params(metric="composite_score")
    logger.info("Best parameters: %s", best_params)

    return best_params


def _count_combinations(param_grid: Dict[str, list]) -> int:
    """Count the total number of hyperparameter combinations."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


# ============================================================================
# EVALUATION & SAVING
# ============================================================================

def evaluate_and_report(
    model: XGBoostGenreClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    genre_names: list,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation and log key metrics.

    Args:
        model: Trained classifier.
        X_test: Test feature matrix.
        y_test: Test labels.
        genre_names: Genre name strings.

    Returns:
        Metrics dictionary from comprehensive_evaluate().
    """
    logger.info("Running comprehensive evaluation...")
    metrics = model.comprehensive_evaluate(X_test, y_test, genre_names)

    conf = metrics["confidence"]
    logger.info(
        "Evaluation results — accuracy: %.4f, f1_macro: %.4f, "
        "f1_weighted: %.4f, composite: %.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
        metrics["composite_score"],
    )
    logger.info(
        "Confidence — correct: %.3f ± %.3f, wrong: %.3f ± %.3f, gap: %.3f",
        conf["confidence_correct_mean"],
        conf["confidence_correct_std"],
        conf["confidence_wrong_mean"],
        conf["confidence_wrong_std"],
        conf["confidence_gap"],
    )
    logger.info(
        "Low-confidence predictions: %d (%.1f%%)",
        conf["low_confidence_count"],
        100 * conf["low_confidence_rate"],
    )

    return metrics


def save_results(
    model: XGBoostGenreClassifier,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
    genre_names: list,
) -> Path:
    """
    Persist model, metrics, and generate analysis.

    Args:
        model: Trained classifier.
        metrics: Comprehensive evaluation metrics.
        args: Parsed command-line arguments.
        genre_names: Genre name strings.

    Returns:
        Path to the saved metrics JSON file.
    """
    model.save()

    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "task": "mono_classification",
        "task_description": "Single-label genre classification (main genre only)",
        "model_type": "xgboost",
        "subset": args.subset,
        "min_samples_per_genre": args.min_samples,
        "use_class_weights": not args.no_weights,
        "params": model.params,
        "metrics": {
            k: v
            for k, v in metrics.items()
            if k not in ("classification_report", "confidence")
        },
        "confidence_analysis": metrics["confidence"],
        "genre_names": genre_names,
        "class_metrics": metrics.get("classification_report"),
    }

    metrics_path = paths.xgboost.metrics_dir / "comprehensive_results.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    logger.info("Metrics saved: %s", metrics_path)

    # Generate visual analysis
    analyzer = ModelAnalyzer(model, genre_names)
    analysis = analyzer.analyze_predictions(X_test=None, y_test=None)
    analyzer.print_analysis_report(analysis)

    return metrics_path


# ============================================================================
# MAIN
# ============================================================================

def main() -> Dict[str, Any]:
    """
    Main entry point for XGBoost training.

    Parses arguments, loads data, trains (or grid-searches),
    evaluates, and persists results.

    Returns:
        Final metrics dictionary.
    """
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(
        level=args.log_level,
        mode="both",
        console_level=args.log_level,
    )

    logger.info("=" * 60)
    logger.info("XGBOOST TRAINING — MONO CLASSIFICATION")
    logger.info("=" * 60)
    logger.info(
        "Config — subset=%s, min_samples=%d, class_weights=%s, "
        "grid_search=%s, grid_size=%s",
        args.subset,
        args.min_samples,
        not args.no_weights,
        args.grid_search,
        args.grid_size if args.grid_search else "n/a",
    )
    logger.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 1. Load data
    data = load_and_report_data(args.subset, args.min_samples)

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    genre_names = data["genre_names"]

    # 2. Grid search or direct training
    if args.grid_search:
        best_params = run_grid_search(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            genre_names,
            args.grid_size,
            not args.no_weights,
        )
        model = XGBoostGenreClassifier(
            params=best_params,
            use_class_weights=not args.no_weights,
        )
    else:
        model = XGBoostGenreClassifier(
            config_path=Path(args.config) if args.config else None,
            use_class_weights=not args.no_weights,
        )

    # 3. Train final model
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        genre_names=genre_names,
    )

    # 4. Evaluate
    metrics = evaluate_and_report(model, X_test, y_test, genre_names)

    # 5. Save
    metrics_path = save_results(model, metrics, args, genre_names)

    logger.info(
        "Training pipeline complete — model: %s, metrics: %s",
        paths.xgboost.models_dir,
        metrics_path,
    )

    return metrics


if __name__ == "__main__":
    main()