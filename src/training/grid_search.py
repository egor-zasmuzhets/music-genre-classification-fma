"""
src/training/grid_search.py
Grid Search для подбора гиперпараметров XGBoost с комплексными метриками
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from sklearn.model_selection import ParameterGrid

from src.data.load_processed import load_data
from src.data.config import paths
from src.models.xgboost_model import XGBoostGenreClassifier


class XGBoostGridSearch:
    """
    Grid Search для XGBoost с комплексными метриками

    Метрики оценки:
    - composite_score (основная): 0.4*F1-macro + 0.4*Top-3 + 0.2*F1-weighted
    - f1_macro: для редких жанров
    - top_3_accuracy: для понимания стилистики
    - accuracy: общая точность
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        use_class_weights: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            param_grid: словарь с сеткой параметров
            use_class_weights: использовать веса классов
            verbose: печатать прогресс
        """
        self.param_grid = param_grid
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        self.results = []

    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        """Генерирует все комбинации параметров"""
        return list(ParameterGrid(self.param_grid))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        genre_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Запускает Grid Search с расширенными метриками
        """
        param_combinations = self._get_param_combinations()
        total = len(param_combinations)

        print("=" * 70)
        print("XGBOOST GRID SEARCH (COMPREHENSIVE METRICS)")
        print("=" * 70)
        print(f"Всего комбинаций: {total}")
        print("Метрики оценки: F1-macro, Top-3 Acc, F1-weighted, Composite Score")
        print("-" * 70)

        for i, params in enumerate(param_combinations, 1):
            if self.verbose:
                print(f"\n[{i}/{total}] Тестирование: {params}")

            start_time = time.time()

            model = XGBoostGenreClassifier(
                params=params,
                use_class_weights=self.use_class_weights
            )

            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                genre_names=genre_names,
                verbose=False
            )

            metrics = model.comprehensive_evaluate(X_test, y_test, genre_names)

            train_time = time.time() - start_time

            result = {
                **params,
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'f1_micro': metrics['f1_micro'],
                'top_1_acc': metrics['top_1_accuracy'],
                'top_3_acc': metrics['top_3_accuracy'],
                'top_5_acc': metrics['top_5_accuracy'],
                'composite_score': metrics['composite_score'],
                'confidence_gap': metrics['confidence']['confidence_gap'],
                'low_confidence_rate': metrics['confidence']['low_confidence_rate'],
                'roc_auc_ovo': metrics.get('roc_auc_ovo', 0),
                'train_time': train_time
            }
            self.results.append(result)

            if self.verbose:
                print(f"  F1-macro: {metrics['f1_macro']:.4f} | "
                      f"Top-3: {metrics['top_3_accuracy']:.4f} | "
                      f"Composite: {metrics['composite_score']:.4f} | "
                      f"Time: {train_time:.1f}s")

        return self.get_results()

    def get_results(self) -> pd.DataFrame:
        """Возвращает результаты в виде DataFrame"""
        df = pd.DataFrame(self.results)
        return df.sort_values('composite_score', ascending=False)

    def get_best_params(self, metric: str = 'composite_score') -> Dict[str, Any]:
        """Возвращает лучшие параметры по выбранной метрике"""
        results = self.get_results()
        best_row = results.loc[results[metric].idxmax()]
        params = {k: best_row[k] for k in self.param_grid.keys()}

        print(f"\n{'='*70}")
        print(f"🏆 ЛУЧШАЯ МОДЕЛЬ ПО {metric.upper()} (MONO CLASSIFICATION)")
        print(f"{'='*70}")
        print("Параметры:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print(f"\nИтоговые метрики:")
        print(f"  Accuracy:      {best_row['accuracy']:.4f}")
        print(f"  F1-macro:      {best_row['f1_macro']:.4f}")
        print(f"  F1-weighted:   {best_row['f1_weighted']:.4f}")
        print(f"  Top-3 Acc:     {best_row['top_3_acc']:.4f}")
        print(f"  Composite:     {best_row['composite_score']:.4f}")
        print(f"{'='*70}")

        return params

    def save_results(self, filepath: Optional[Path] = None):
        """Сохраняет результаты в CSV"""
        df = self.get_results()

        if filepath is None:
            filepath = paths.xgboost_mono_grid_search_dir / "grid_search_results.csv"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Результаты Grid Search сохранены: {filepath}")

    def get_best_by_metric(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает лучшие модели по разным метрикам"""
        results = self.get_results()

        return {
            'by_composite': {
                'params': {k: results.iloc[0][k] for k in self.param_grid.keys()},
                'metrics': results.iloc[0].to_dict()
            },
            'by_f1_macro': {
                'params': {k: results.loc[results['f1_macro'].idxmax(), k] for k in self.param_grid.keys()},
                'metrics': results.loc[results['f1_macro'].idxmax()].to_dict()
            },
            'by_top_3': {
                'params': {k: results.loc[results['top_3_acc'].idxmax(), k] for k in self.param_grid.keys()},
                'metrics': results.loc[results['top_3_acc'].idxmax()].to_dict()
            },
            'by_accuracy': {
                'params': {k: results.loc[results['accuracy'].idxmax(), k] for k in self.param_grid.keys()},
                'metrics': results.loc[results['accuracy'].idxmax()].to_dict()
            }
        }


# Предопределённые сетки параметров
GRID_SMALL = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.3]
}

GRID_MEDIUM = {
    'max_depth': [3, 5, 7, 9],
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.1, 0.2, 0.3],
    'subsample': [0.8, 1.0],
    'min_child_weight': [1, 3]
}

GRID_FULL = {
    'max_depth': [3, 5, 7, 9, 11],
    'n_estimators': [50, 100, 150, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 1.5]
}