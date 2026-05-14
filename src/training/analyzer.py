"""
src/training/analyzer.py
Model analysis — metrics visualization, error analysis, rare genre diagnostics.

Provides comprehensive post-training analysis for XGBoost classifiers:
confusion matrix plots, per-class F1 scores, confidence distributions,
Top-k accuracy curves, and rare vs. common genre quality comparison.

Typical usage:
    from src.training.analyzer import ModelAnalyzer

    analyzer = ModelAnalyzer(model, genre_names)
    results = analyzer.analyze_predictions(X_test, y_test)
    analyzer.print_analysis_report(results)
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.utils.config import paths


logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """
    Post-training analysis and visualization for genre classifiers.

    Generates a suite of diagnostic plots and reports covering:
    - Normalized confusion matrix
    - Per-class F1 scores (with sample counts)
    - Model confidence distribution (correct vs. incorrect)
    - Top-k accuracy curve
    - Rare vs. common genre quality gap

    All plots can be saved to disk or displayed interactively.

    Attributes:
        model: A fitted classifier with predict() and predict_proba().
        genre_names: Ordered list of genre name strings.
    """

    def __init__(
        self,
        model: Any,
        genre_names: List[str],
    ):
        """
        Initialize the analyzer.

        Args:
            model: Fitted classifier. Must implement predict(), predict_proba(),
                   and comprehensive_evaluate() or top_k_accuracy().
            genre_names: Ordered list of genre names matching class indices.
        """
        self.model = model
        self.genre_names = genre_names

        logger.debug(
            "ModelAnalyzer initialized — %d genres",
            len(genre_names),
        )

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def analyze_predictions(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete analysis suite and generate all plots.

        Args:
            X_test: Test feature matrix.
            y_test: Ground truth labels.
            save_dir: Directory for saving plot files.
                      Defaults to paths.xgboost.plots_dir.

        Returns:
            Analysis results dictionary with keys:
            - metrics: comprehensive evaluation metrics
            - per_class: per-genre classification report
            - misclassified_count, misclassified_rate
            - confidence_correct, confidence_wrong
            - confusion_pairs: top confused genre pairs
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        metrics = self.model.comprehensive_evaluate(
            X_test, y_test, self.genre_names
        )

        report = classification_report(
            y_test, y_pred,
            target_names=self.genre_names,
            output_dict=True,
            zero_division=0,
        )

        misclassified_idx = np.where(y_pred != y_test)[0]
        correct_idx = np.where(y_pred == y_test)[0]

        max_proba = y_pred_proba.max(axis=1)
        confidence_correct = (
            float(max_proba[correct_idx].mean()) if len(correct_idx) > 0 else 0.0
        )
        confidence_wrong = (
            float(max_proba[misclassified_idx].mean()) if len(misclassified_idx) > 0 else 0.0
        )

        confusion_pairs = self._analyze_confusions(y_test, y_pred)

        results = {
            "metrics": metrics,
            "per_class": report,
            "misclassified_count": len(misclassified_idx),
            "misclassified_rate": (
                len(misclassified_idx) / len(y_test) if len(y_test) > 0 else 0.0
            ),
            "confidence_correct": confidence_correct,
            "confidence_wrong": confidence_wrong,
            "confusion_pairs": confusion_pairs,
        }

        if save_dir is None:
            save_dir = paths.xgboost.plots_dir
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating analysis plots in: %s", save_dir)

        self.plot_confusion_matrix(y_test, y_pred, save_dir / "confusion_matrix.png")
        self.plot_per_class_f1(report, save_dir / "per_class_f1.png")
        self.plot_confidence_distribution(
            max_proba, y_pred == y_test, save_dir / "confidence_dist.png"
        )
        self.plot_topk_accuracy(
            y_test, y_pred_proba, save_path=save_dir / "topk_accuracy.png"
        )
        self.plot_rare_genres_analysis(
            y_test, y_pred, save_path=save_dir / "rare_genres_analysis.png"
        )

        logger.info("Analysis complete — %d plots saved", 5)

        return results

    def _analyze_confusions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> List[Tuple[str, str, int]]:
        """
        Extract the most frequently confused genre pairs.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Top-10 list of (true_genre, predicted_genre, count) tuples,
            sorted by count descending.
        """
        cm = confusion_matrix(y_true, y_pred)

        confusions = []
        for i in range(len(self.genre_names)):
            for j in range(len(self.genre_names)):
                if i != j and cm[i, j] > 0:
                    confusions.append(
                        (self.genre_names[i], self.genre_names[j], int(cm[i, j]))
                    )

        return sorted(confusions, key=lambda x: x[2], reverse=True)[:10]

    # ------------------------------------------------------------------
    # Plot: Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot a normalized confusion matrix heatmap.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            save_path: If provided, saves the figure to this path.
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.genre_names,
            yticklabels=self.genre_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(
            "Confusion Matrix — Single-Label Genre Classification (normalized)",
            fontsize=14,
        )
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.debug("Confusion matrix saved: %s", save_path.name)

        plt.show()

    # ------------------------------------------------------------------
    # Plot: Per-Class F1
    # ------------------------------------------------------------------

    def plot_per_class_f1(
        self,
        report: Dict[str, Any],
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot horizontal bar chart of F1 scores per genre.

        Bar color intensity reflects score; annotations include sample count.

        Args:
            report: Classification report dict (output_dict=True).
            save_path: If provided, saves the figure to this path.
        """
        f1_scores = [
            report[genre]["f1-score"] for genre in self.genre_names
        ]
        supports = [
            report[genre]["support"] for genre in self.genre_names
        ]

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(self.genre_names)))
        bars = ax.barh(self.genre_names, f1_scores, color=colors)
        ax.set_xlabel("F1-score", fontsize=12)
        ax.set_title("F1-score by Genre (bar label = sample count)", fontsize=14)
        ax.set_xlim(0, 1)

        for bar, score, support in zip(bars, f1_scores, supports):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f} (n={support})",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.debug("Per-class F1 plot saved: %s", save_path.name)

        plt.show()

    # ------------------------------------------------------------------
    # Plot: Confidence Distribution
    # ------------------------------------------------------------------

    def plot_confidence_distribution(
        self,
        confidences: np.ndarray,
        is_correct: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot overlapping histograms of max probability for correct
        and incorrect predictions.

        Args:
            confidences: 1D array of max predicted probabilities per sample.
            is_correct: 1D boolean array (True if prediction was correct).
            save_path: If provided, saves the figure to this path.
        """
        correct_conf = confidences[is_correct]
        wrong_conf = confidences[~is_correct]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            correct_conf, bins=20, alpha=0.7,
            label=f"Correct (n={len(correct_conf)})", color="green",
        )
        ax.hist(
            wrong_conf, bins=20, alpha=0.7,
            label=f"Wrong (n={len(wrong_conf)})", color="red",
        )
        ax.axvline(x=0.5, color="gray", linestyle="--", label="Threshold (0.5)")
        ax.set_xlabel("Max Probability", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Model Confidence Distribution", fontsize=14)
        ax.legend()

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.debug("Confidence distribution saved: %s", save_path.name)

        plt.show()

    # ------------------------------------------------------------------
    # Plot: Top-k Accuracy
    # ------------------------------------------------------------------

    def plot_topk_accuracy(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        max_k: int = 10,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot Top-k accuracy as a function of k.

        Args:
            y_true: Ground truth labels.
            y_pred_proba: Predicted probability matrix.
            max_k: Maximum k value to plot (default: 10).
            save_path: If provided, saves the figure to this path.
        """
        k_values = list(range(1, max_k + 1))
        accuracies = [
            self.model.top_k_accuracy(y_true, y_pred_proba, k=k)
            for k in k_values
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, accuracies, "bo-", linewidth=2, markersize=8)
        ax.fill_between(k_values, accuracies, alpha=0.2)
        ax.set_xlabel("k", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Top-k Accuracy — Single-Label Classification", fontsize=14)
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        for k, acc in zip(k_values, accuracies):
            ax.annotate(
                f"{acc:.3f}", (k, acc),
                textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.debug("Top-k accuracy plot saved: %s", save_path.name)

        plt.show()

    # ------------------------------------------------------------------
    # Plot: Rare vs. Common Genre Analysis
    # ------------------------------------------------------------------

    def plot_rare_genres_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        min_support: int = 50,
        save_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Compare model quality on rare vs. common genres.

        Genres with ≤ min_support test samples are considered rare.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            min_support: Maximum samples for a genre to be considered rare.
            save_path: If provided, saves the figure to this path.

        Returns:
            Dictionary with keys 'rare_f1', 'common_f1', 'gap'.
        """
        genre_counts = Counter(y_true)

        rare_genres = [
            genre for i, genre in enumerate(self.genre_names)
            if genre_counts.get(i, 0) <= min_support
        ]
        common_genres = [
            genre for i, genre in enumerate(self.genre_names)
            if genre_counts.get(i, 0) > min_support
        ]

        def get_mask(genres_list):
            mask = np.zeros(len(y_true), dtype=bool)
            for g in genres_list:
                idx = self.genre_names.index(g)
                mask |= y_true == idx
            return mask

        mask_rare = get_mask(rare_genres)
        mask_common = get_mask(common_genres)

        f1_rare = f1_score(
            y_true[mask_rare], y_pred[mask_rare],
            average="weighted", zero_division=0,
        )
        f1_common = f1_score(
            y_true[mask_common], y_pred[mask_common],
            average="weighted", zero_division=0,
        )
        gap = f1_common - f1_rare

        logger.info(
            "Rare genre analysis — rare: %d genres (F1=%.4f), "
            "common: %d genres (F1=%.4f), gap: %.4f",
            len(rare_genres), f1_rare,
            len(common_genres), f1_common,
            gap,
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ["Rare genres", "Common genres"]
        scores = [f1_rare, f1_common]
        colors = ["#e74c3c", "#2ecc71"]

        bars = ax.bar(categories, scores, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1-score (weighted)", fontsize=12)
        ax.set_title(
            f"Quality on Rare (≤{min_support}) vs Common Genres",
            fontsize=14,
        )

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{score:.4f}",
                ha="center",
                fontsize=11,
            )

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.debug("Rare genre analysis saved: %s", save_path.name)

        plt.show()

        return {"rare_f1": f1_rare, "common_f1": f1_common, "gap": gap}

    # ------------------------------------------------------------------
    # Report printing
    # ------------------------------------------------------------------

    def print_analysis_report(self, results: Dict[str, Any]) -> None:
        """
        Print a formatted analysis report to stdout.

        This is a manual debugging/exploration utility — not logged.

        Args:
            results: Results dictionary from analyze_predictions().
        """
        metrics = results["metrics"]

        print("=" * 70)
        print("COMPREHENSIVE MODEL ANALYSIS — MONO CLASSIFICATION")
        print("=" * 70)
        print(f"Accuracy:            {metrics['accuracy']:.4f}")
        print(f"F1-macro:            {metrics['f1_macro']:.4f}")
        print(f"F1-weighted:         {metrics['f1_weighted']:.4f}")
        print(f"F1-micro:            {metrics['f1_micro']:.4f}")
        print("-" * 70)
        print(f"Top-1 Accuracy:      {metrics.get('top_1_accuracy', 'N/A'):.4f}")
        print(f"Top-3 Accuracy:      {metrics.get('top_3_accuracy', 'N/A'):.4f}")
        print(f"Top-5 Accuracy:      {metrics.get('top_5_accuracy', 'N/A'):.4f}")
        print("-" * 70)
        print(f"Composite Score:     {metrics['composite_score']:.4f}")
        print(f"ROC-AUC (ovo):       {metrics.get('roc_auc_ovo', 'N/A')}")
        print(f"ROC-AUC (ovr):       {metrics.get('roc_auc_ovr', 'N/A')}")
        print("-" * 70)
        print(
            f"Misclassified:       {results['misclassified_count']} "
            f"({results['misclassified_rate']:.2%})"
        )
        print(f"Confidence (correct): {results['confidence_correct']:.3f}")
        print(f"Confidence (wrong):   {results['confidence_wrong']:.3f}")
        gap = results["confidence_correct"] - results["confidence_wrong"]
        print(f"Confidence gap:       {gap:.3f}")
        print("-" * 70)
        print("\nTop-10 most confused genre pairs (true → predicted):")
        for true_genre, pred_genre, count in results["confusion_pairs"][:10]:
            print(f"  {true_genre:20s} → {pred_genre:20s}: {count}×")
        print("=" * 70)