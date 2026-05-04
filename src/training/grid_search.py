"""
src/training/grid_search.py
Grid Search для подбора гиперпараметров XGBoost с комплексными метриками
"""

import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime
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

    # Имя файла по умолчанию для сохранения результатов
    DEFAULT_RESULTS_NAME = "grid_search_results.csv"

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        use_class_weights: bool = True,
        verbose: bool = True,
        results_name: Optional[str] = None
    ):
        """
        Args:
            param_grid: словарь с сеткой параметров
            use_class_weights: использовать веса классов
            verbose: печатать прогресс
            results_name: имя файла для сохранения результатов (без расширения)
        """
        self.param_grid = param_grid
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        self.results_name = results_name or self.DEFAULT_RESULTS_NAME
        self.results = []
        self.best_model_info = None

    def _get_default_path(self, filename: Optional[str] = None) -> Path:
        """
        Возвращает путь по умолчанию для результатов

        Args:
            filename: имя файла (если None — используется self.results_name)
        """
        name = filename or self.results_name
        # Убеждаемся, что имя имеет расширение .csv
        if not name.endswith('.csv'):
            name = name + '.csv'
        return paths.xgboost_mono_grid_search_dir / name

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
        genre_names: Optional[List[str]] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Запускает Grid Search с расширенными метриками

        Args:
            save_intermediate: сохранять промежуточные результаты после каждой итерации
        """
        param_combinations = self._get_param_combinations()
        total = len(param_combinations)

        print("=" * 70)
        print("XGBOOST GRID SEARCH (COMPREHENSIVE METRICS)")
        print("=" * 70)
        print(f"Всего комбинаций: {total}")
        print("Метрики оценки: F1-macro, Top-3 Acc, F1-weighted, Composite Score")
        print("-" * 70)

        best_composite = -1
        best_f1_macro = -1
        best_top3 = -1
        best_accuracy = -1

        best_composite_params = None
        best_f1_macro_params = None
        best_top3_params = None
        best_accuracy_params = None

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

            # Отслеживаем лучшие модели
            if metrics['composite_score'] > best_composite:
                best_composite = metrics['composite_score']
                best_composite_params = params.copy()

            if metrics['f1_macro'] > best_f1_macro:
                best_f1_macro = metrics['f1_macro']
                best_f1_macro_params = params.copy()

            if metrics['top_3_accuracy'] > best_top3:
                best_top3 = metrics['top_3_accuracy']
                best_top3_params = params.copy()

            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_accuracy_params = params.copy()

            if self.verbose:
                print(f"  F1-macro: {metrics['f1_macro']:.4f} | "
                      f"Top-3: {metrics['top_3_accuracy']:.4f} | "
                      f"Composite: {metrics['composite_score']:.4f} | "
                      f"Time: {train_time:.1f}s")

            # Сохраняем промежуточные результаты
            if save_intermediate and i % 5 == 0:
                self.save_results()
                print(f"  📁 Промежуточные результаты сохранены ({i}/{total})")

        # Сохраняем информацию о лучших моделях
        self.best_model_info = {
            'by_composite': {'params': best_composite_params, 'score': best_composite},
            'by_f1_macro': {'params': best_f1_macro_params, 'score': best_f1_macro},
            'by_top_3': {'params': best_top3_params, 'score': best_top3},
            'by_accuracy': {'params': best_accuracy_params, 'score': best_accuracy},
            'total_combinations': total,
            'completed_at': datetime.now().isoformat()
        }

        # Сохраняем финальные результаты
        self.save_results()

        return self.get_results()

    def get_results(self) -> pd.DataFrame:
        """Возвращает результаты в виде DataFrame"""
        df = pd.DataFrame(self.results)
        if len(df) > 0:
            return df.sort_values('composite_score', ascending=False)
        return df

    def get_best_params(self, metric: str = 'composite_score') -> Dict[str, Any]:
        """Возвращает лучшие параметры по выбранной метрике"""
        results = self.get_results()
        if len(results) == 0:
            raise ValueError("Нет результатов. Сначала запустите fit()")

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

    def save_results(self, filepath: Optional[Path] = None, name: Optional[str] = None) -> Path:
        """
        Сохраняет результаты в CSV

        Args:
            filepath: полный путь к файлу (если указан, приоритет выше)
            name: имя файла (сохраняется в директорию по умолчанию)

        Returns:
            Путь к сохранённому файлу
        """
        df = self.get_results()

        if len(df) == 0:
            print("⚠️ Нет результатов для сохранения")
            return None

        # Определяем путь для сохранения
        if filepath is not None:
            save_path = Path(filepath)
        elif name is not None:
            save_path = self._get_default_path(name)
        else:
            save_path = self._get_default_path()

        save_path = Path(save_path)
        if save_path.suffix != '.csv':
            save_path = save_path.with_suffix('.csv')

        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"✅ Результаты Grid Search сохранены: {save_path}")

        # Также сохраняем JSON с метаданными
        meta_path = save_path.with_suffix('.meta.json')
        if self.best_model_info:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.best_model_info, f, indent=2, ensure_ascii=False)
            print(f"✅ Метаданные сохранены: {meta_path}")

        return save_path

    def load_results(self, filepath: Optional[Path] = None, name: Optional[str] = None) -> pd.DataFrame:
        """
        Загружает результаты из CSV

        Args:
            filepath: полный путь к файлу (если указан, приоритет выше)
            name: имя файла (загружается из директории по умолчанию)

        Returns:
            DataFrame с результатами
        """
        # Определяем путь для загрузки
        if filepath is not None:
            load_path = Path(filepath)
        elif name is not None:
            load_path = self._get_default_path(name)
        else:
            load_path = self._get_default_path()

        load_path = Path(load_path)

        if not load_path.exists():
            # Пробуем без расширения
            alt_path = load_path.with_suffix('')
            if alt_path.exists():
                load_path = alt_path
            else:
                raise FileNotFoundError(f"Файл не найден: {load_path}")

        df = pd.read_csv(load_path)
        self.results = df.to_dict('records')

        print(f"✅ Результаты загружены: {load_path}")
        print(f"   Всего записей: {len(df)}")

        return df

    def get_best_by_metric(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает лучшие модели по разным метрикам"""
        results = self.get_results()
        if len(results) == 0:
            return {}

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

    def print_summary(self):
        """Выводит сводку о результатах Grid Search"""
        if len(self.results) == 0:
            print("Нет результатов. Сначала запустите fit()")
            return

        results_df = self.get_results()

        print("=" * 70)
        print("СВОДКА GRID SEARCH")
        print("=" * 70)
        print(f"Всего комбинаций: {len(self.results)}")
        print(f"Лучший composite_score: {results_df.iloc[0]['composite_score']:.4f}")
        print(f"Лучший F1-macro: {results_df['f1_macro'].max():.4f}")
        print(f"Лучший Top-3: {results_df['top_3_acc'].max():.4f}")
        print(f"Лучший Accuracy: {results_df['accuracy'].max():.4f}")
        print("-" * 70)

        if self.best_model_info:
            print("\nЛучшие параметры по каждой метрике:")
            for metric, info in self.best_model_info.items():
                if info['params']:
                    print(f"  {metric}: score={info['score']:.4f}")
        print("=" * 70)


GRID_TEST = {
    'max_depth': [3, 5],
    'n_estimators': [50, 100]
}

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