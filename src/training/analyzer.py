"""
src/training/analyzer.py
Анализ модели: метрики, визуализация, ошибки, комплексная оценка
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from src.data.config import paths
from src.models.xgboost_model import XGBoostGenreClassifier


class ModelAnalyzer:
    """
    Анализ обученной модели XGBoost

    Позволяет:
    - Визуализировать confusion matrix
    - Анализировать ошибки предсказаний
    - Выявлять проблемные жанры
    - Генерировать отчёты
    - Анализировать редкие жанры
    - Строить графики Top-k Accuracy
    """

    def __init__(
        self,
        model: XGBoostGenreClassifier,
        genre_names: List[str]
    ):
        self.model = model
        self.genre_names = genre_names

    def analyze_predictions(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Полный анализ предсказаний модели"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        metrics = self.model.comprehensive_evaluate(X_test, y_test, self.genre_names)

        report = classification_report(
            y_test, y_pred, target_names=self.genre_names, output_dict=True, zero_division=0
        )

        misclassified_idx = np.where(y_pred != y_test)[0]
        correct_idx = np.where(y_pred == y_test)[0]

        max_proba = y_pred_proba.max(axis=1)
        confidence_correct = max_proba[correct_idx].mean() if len(correct_idx) > 0 else 0
        confidence_wrong = max_proba[misclassified_idx].mean() if len(misclassified_idx) > 0 else 0

        confusion_pairs = self._analyze_confusions(y_test, y_pred)

        results = {
            'metrics': metrics,
            'per_class': report,
            'misclassified_count': len(misclassified_idx),
            'misclassified_rate': len(misclassified_idx) / len(y_test),
            'confidence_correct': confidence_correct,
            'confidence_wrong': confidence_wrong,
            'confusion_pairs': confusion_pairs
        }

        if save_dir is None:
            save_dir = paths.xgboost_mono_plots_dir
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        self.plot_confusion_matrix(y_test, y_pred, save_dir / 'confusion_matrix.png')
        self.plot_per_class_f1(report, save_dir / 'per_class_f1.png')
        self.plot_confidence_distribution(max_proba, y_pred == y_test, save_dir / 'confidence_dist.png')
        self.plot_topk_accuracy(y_test, y_pred_proba, save_path=save_dir / 'topk_accuracy.png')
        self.plot_rare_genres_analysis(y_test, y_pred, save_path=save_dir / 'rare_genres_analysis.png')

        return results

    def _analyze_confusions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> List[Tuple[str, str, int]]:
        """Анализирует, какие жанры с чем путаются"""
        cm = confusion_matrix(y_true, y_pred)

        confusions = []
        for i in range(len(self.genre_names)):
            for j in range(len(self.genre_names)):
                if i != j and cm[i, j] > 0:
                    confusions.append((self.genre_names[i], self.genre_names[j], int(cm[i, j])))

        return sorted(confusions, key=lambda x: x[2], reverse=True)[:10]

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """Строит и сохраняет confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.genre_names, yticklabels=self.genre_names, ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix - Single-Label Genre Classification (normalized)', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_per_class_f1(self, report: Dict, save_path: Optional[Path] = None):
        """Строит график F1-score по каждому жанру"""
        f1_scores = [report[genre]['f1-score'] for genre in self.genre_names]
        supports = [report[genre]['support'] for genre in self.genre_names]

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(self.genre_names)))
        bars = ax.barh(self.genre_names, f1_scores, color=colors)
        ax.set_xlabel('F1-score', fontsize=12)
        ax.set_title('F1-score by Genre (size ~ number of samples)', fontsize=14)
        ax.set_xlim(0, 1)

        for bar, score, support in zip(bars, f1_scores, supports):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f} (n={support})', va='center', fontsize=9)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_confidence_distribution(
        self,
        confidences: np.ndarray,
        is_correct: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """Строит распределение уверенности модели"""
        fig, ax = plt.subplots(figsize=(10, 6))

        correct_conf = confidences[is_correct]
        wrong_conf = confidences[~is_correct]

        ax.hist(correct_conf, bins=20, alpha=0.7,
                label=f'Correct (n={len(correct_conf)})', color='green')
        ax.hist(wrong_conf, bins=20, alpha=0.7,
                label=f'Wrong (n={len(wrong_conf)})', color='red')
        ax.axvline(x=0.5, color='gray', linestyle='--', label='Threshold (0.5)')
        ax.set_xlabel('Max Probability', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Model Confidence Distribution', fontsize=14)
        ax.legend()

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_topk_accuracy(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        max_k: int = 10,
        save_path: Optional[Path] = None
    ):
        """Строит график зависимости Top-k Accuracy от k"""
        k_values = list(range(1, max_k + 1))
        accuracies = []

        for k in k_values:
            acc = self.model.top_k_accuracy(y_true, y_pred_proba, k=k)
            accuracies.append(acc)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
        ax.fill_between(k_values, accuracies, alpha=0.2)
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Top-k Accuracy - Single-Label Classification', fontsize=14)
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        for k, acc in zip(k_values, accuracies):
            ax.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_rare_genres_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        min_support: int = 50,
        save_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """Анализирует качество на редких жанрах"""
        from collections import Counter

        genre_counts = Counter(y_true)

        rare_genres = []
        common_genres = []

        for i, genre in enumerate(self.genre_names):
            if genre_counts.get(i, 0) <= min_support:
                rare_genres.append(genre)
            else:
                common_genres.append(genre)

        def get_mask(genres_list):
            mask = np.zeros(len(y_true), dtype=bool)
            for g in genres_list:
                idx = self.genre_names.index(g)
                mask |= (y_true == idx)
            return mask

        mask_rare = get_mask(rare_genres)
        mask_common = get_mask(common_genres)

        f1_rare = f1_score(y_true[mask_rare], y_pred[mask_rare], average='weighted', zero_division=0)
        f1_common = f1_score(y_true[mask_common], y_pred[mask_common], average='weighted', zero_division=0)

        print("=" * 60)
        print("АНАЛИЗ РЕДКИХ ЖАНРОВ")
        print("=" * 60)
        print(f"Редкие жанры (≤ {min_support} треков): {len(rare_genres)}")
        for g in rare_genres:
            count = genre_counts.get(self.genre_names.index(g), 0)
            print(f"  - {g}: {count} треков")
        print(f"\nF1 на редких жанрах: {f1_rare:.4f}")
        print(f"\nЧастые жанры (> {min_support} треков): {len(common_genres)}")
        print(f"F1 на частых жанрах: {f1_common:.4f}")
        print(f"\nРазрыв: {f1_common - f1_rare:.4f}")
        print("=" * 60)

        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['Редкие жанры', 'Частые жанры']
        scores = [f1_rare, f1_common]
        colors = ['#e74c3c', '#2ecc71']

        bars = ax.bar(categories, scores, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('F1-score (weighted)', fontsize=12)
        ax.set_title('Quality on Rare vs Common Genres', fontsize=14)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.4f}', ha='center', fontsize=11)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return {'rare_f1': f1_rare, 'common_f1': f1_common, 'gap': f1_common - f1_rare}

    def print_analysis_report(self, results: Dict[str, Any]):
        """Печатает отчёт об анализе"""
        metrics = results['metrics']

        print("=" * 70)
        print("КОМПЛЕКСНЫЙ АНАЛИЗ МОДЕЛИ XGBOOST (MONO CLASSIFICATION)")
        print("=" * 70)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"F1-macro:          {metrics['f1_macro']:.4f}")
        print(f"F1-weighted:       {metrics['f1_weighted']:.4f}")
        print(f"F1-micro:          {metrics['f1_micro']:.4f}")
        print("-" * 70)
        print(f"Top-1 Accuracy:    {metrics['top_1_accuracy']:.4f}")
        print(f"Top-3 Accuracy:    {metrics['top_3_accuracy']:.4f}")
        print(f"Top-5 Accuracy:    {metrics['top_5_accuracy']:.4f}")
        print("-" * 70)
        print(f"Composite Score:   {metrics['composite_score']:.4f}")
        print(f"ROC-AUC (ovo):     {metrics.get('roc_auc_ovo', 'N/A')}")
        print("-" * 70)
        print(f"Ошибок:            {results['misclassified_count']} ({results['misclassified_rate']:.2%})")
        print(f"Уверенность (прав.): {results['confidence_correct']:.3f}")
        print(f"Уверенность (ошиб.): {results['confidence_wrong']:.3f}")
        print(f"Разрыв уверенности:  {results['confidence_correct'] - results['confidence_wrong']:.3f}")
        print("-" * 70)
        print("\nТоп-10 частых ошибок (жанр → предсказание):")
        for true_genre, pred_genre, count in results['confusion_pairs'][:10]:
            print(f"  {true_genre:20s} → {pred_genre:20s}: {count} раз")
        print("=" * 70)