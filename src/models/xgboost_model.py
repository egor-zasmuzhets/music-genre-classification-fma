"""
src/models/xgboost_model.py
XGBoost модель для классификации музыкальных жанров с комплексной оценкой
"""

import json
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

from src.data.config import paths as project_paths
from src.data.load_processed import load_data


class XGBoostGenreClassifier:
    """
    XGBoost классификатор для музыкальных жанров.

    Поддерживает:
    - Обучение с/без весов классов
    - Early stopping на валидации
    - Сохранение/загрузку модели
    - Комплексную оценку (Top-k, Confidence, Composite Score)
    """

    # Имя файла по умолчанию для сохранения модели
    DEFAULT_MODEL_NAME = "xgboost_auto.json"

    def __init__(
        self,
        config_path: Optional[Path] = None,
        params: Optional[Dict[str, Any]] = None,
        use_class_weights: bool = True,
        random_state: int = 42,
        model_name: Optional[str] = None
    ):
        """
        Args:
            config_path: путь к YAML конфигу (если None — стандартный)
            params: параметры XGBoost (если указаны, переопределяют конфиг)
            use_class_weights: использовать ли веса классов
            random_state: seed для воспроизводимости
            model_name: имя файла для сохранения/загрузки модели
        """
        # Загружаем конфиг
        if config_path is None:
            config_path = project_paths.configs_dir / "model_config.yaml"

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {'parameters': {}, 'training': {}}

        # Параметры модели
        self.params = params or self.config.get('parameters', {}).copy()
        self.params['random_state'] = random_state
        self.params['objective'] = 'multi:softmax'

        self.use_class_weights = use_class_weights
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model: Optional[xgb.Booster] = None
        self.sklearn_model: Optional[xgb.XGBClassifier] = None
        self._is_fitted = False
        self.best_iteration: Optional[int] = None
        self.class_weights: Optional[Dict[int, float]] = None
        self.genre_names: Optional[List[str]] = None

    def _get_default_path(self, filename: Optional[str] = None) -> Path:
        """
        Возвращает путь по умолчанию для модели

        Args:
            filename: имя файла (если None — используется self.model_name)
        """
        name = filename or self.model_name
        # Убеждаемся, что имя имеет расширение .json
        if not name.endswith('.json'):
            name = name + '.json'
        return project_paths.xgboost_mono_models_dir / name

    def _get_sample_weights(self, y_train: np.ndarray) -> Optional[np.ndarray]:
        """Вычисляет веса для каждого трека"""
        if not self.use_class_weights:
            return None

        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        self.class_weights = dict(zip(classes, weights))

        sample_weights = np.array([self.class_weights[y] for y in y_train])
        return sample_weights

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        genre_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> 'XGBoostGenreClassifier':
        """Обучает модель"""
        self.genre_names = genre_names
        num_classes = len(np.unique(y_train))

        self.params['num_class'] = num_classes
        sample_weights = self._get_sample_weights(y_train)

        if verbose:
            print("=" * 60)
            print("Обучение XGBoost классификатора")
            print("=" * 60)
            print(f"Треков train: {len(X_train)}")
            print(f"Признаков: {X_train.shape[1]}")
            print(f"Классов: {num_classes}")
            print(f"Веса классов: {self.use_class_weights}")
            print(f"Параметры: {self.params}")
            print("-" * 60)

        self.sklearn_model = xgb.XGBClassifier(
            **self.params,
            eval_metric='mlogloss'
        )

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.sklearn_model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=verbose
        )

        self.model = self.sklearn_model.get_booster()
        self._is_fitted = True

        if verbose:
            print("-" * 60)
            print("✅ Обучение завершено")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает классы"""
        if not self._is_fitted:
            raise ValueError("Модель не обучена")
        return self.sklearn_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает вероятности классов"""
        if not self._is_fitted:
            raise ValueError("Модель не обучена")
        return self.sklearn_model.predict_proba(X)

    def top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        k: int = 3
    ) -> float:
        """Рассчитывает Top-k Accuracy"""
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

    def confidence_analysis(
        self,
        X_test: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Анализирует уверенность модели в предсказаниях"""
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)

        max_proba = y_pred_proba.max(axis=1)
        is_correct = (y_pred == y_true)

        correct_conf = max_proba[is_correct]
        wrong_conf = max_proba[~is_correct]

        return {
            'confidence_correct_mean': float(correct_conf.mean()) if len(correct_conf) > 0 else 0,
            'confidence_correct_std': float(correct_conf.std()) if len(correct_conf) > 0 else 0,
            'confidence_wrong_mean': float(wrong_conf.mean()) if len(wrong_conf) > 0 else 0,
            'confidence_wrong_std': float(wrong_conf.std()) if len(wrong_conf) > 0 else 0,
            'confidence_gap': float(correct_conf.mean() - wrong_conf.mean()) if len(wrong_conf) > 0 else 0,
            'low_confidence_count': int(np.sum(max_proba < 0.5)),
            'low_confidence_rate': float(np.mean(max_proba < 0.5))
        }

    def _compute_composite_score(self, metrics: Dict[str, Any]) -> float:
        """
        Вычисляет композитный скор для выбора лучшей модели

        Веса:
        - F1-macro: 40% (редкие жанры)
        - Top-3 accuracy: 40% (понимание стилистики)
        - F1-weighted: 20% (общее качество)
        """
        f1_macro = metrics.get('f1_macro', 0)
        top_3 = metrics.get('top_3_accuracy', 0)
        f1_weighted = metrics.get('f1_weighted', 0)

        return 0.4 * f1_macro + 0.4 * top_3 + 0.2 * f1_weighted

    def comprehensive_evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        genre_names: Optional[List[str]] = None,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, Any]:
        """Комплексная оценка модели со всеми метриками"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        target_names = genre_names or self.genre_names

        # Базовые метрики
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            'f1_micro': float(f1_score(y_test, y_pred, average='micro', zero_division=0)),
        }

        # Top-k метрики
        for k in k_values:
            top_k = self.top_k_accuracy(y_test, y_pred_proba, k=k)
            metrics[f'top_{k}_accuracy'] = float(top_k)

        # Confidence analysis
        metrics['confidence'] = self.confidence_analysis(X_test, y_test)

        # Classification report
        if target_names:
            metrics['classification_report'] = classification_report(
                y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
            )

        # ROC-AUC
        try:
            metrics['roc_auc_ovo'] = float(roc_auc_score(
                y_test, y_pred_proba, multi_class='ovo', average='weighted'
            ))
            metrics['roc_auc_ovr'] = float(roc_auc_score(
                y_test, y_pred_proba, multi_class='ovr', average='weighted'
            ))
        except:
            metrics['roc_auc_ovo'] = None
            metrics['roc_auc_ovr'] = None

        # Composite score
        metrics['composite_score'] = self._compute_composite_score(metrics)

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Возвращает важность признаков"""
        if not self._is_fitted:
            raise ValueError("Модель не обучена")

        importance = self.sklearn_model.feature_importances_

        try:
            data = load_data()
            feature_names = data['metadata'].get('feature_names')
            if feature_names is None or len(feature_names) != len(importance):
                feature_names = [f'feature_{i}' for i in range(len(importance))]
        except:
            feature_names = [f'feature_{i}' for i in range(len(importance))]

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df.head(top_n)

    def save(self, filepath: Optional[Path] = None, name: Optional[str] = None) -> Path:
        """
        Сохраняет модель и её конфигурацию

        Args:
            filepath: полный путь к файлу (если указан, приоритет выше)
            name: имя файла (сохраняется в директорию по умолчанию)

        Returns:
            Путь к сохранённому файлу

        Examples:
            # Сохранить с именем по умолчанию (xgboost_auto.json)
            model.save()

            # Сохранить с указанным именем
            model.save(name="my_best_model")

            # Сохранить по полному пути
            model.save(filepath=Path("./custom/path/model.json"))

            # Сохранить с именем в директорию по умолчанию
            model.save(name="experiment_1")
        """
        if not self._is_fitted:
            raise ValueError("Нечего сохранять — модель не обучена")

        # Определяем путь для сохранения
        if filepath is not None:
            # Используем полный путь
            save_path = Path(filepath)
        elif name is not None:
            # Используем имя в директории по умолчанию
            save_path = self._get_default_path(name)
        else:
            # Используем имя по умолчанию
            save_path = self._get_default_path()

        save_path = Path(save_path)
        # Убеждаемся, что расширение .json
        if save_path.suffix != '.json':
            save_path = save_path.with_suffix('.json')

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем модель
        self.sklearn_model.save_model(str(save_path))

        # Сохраняем метаданные
        meta_path = save_path.with_suffix('.meta.json')
        meta = {
            'params': self.params,
            'use_class_weights': self.use_class_weights,
            'genre_names': self.genre_names,
            'class_weights': {int(k): float(v) for k, v in self.class_weights.items()} if self.class_weights else None,
            'is_fitted': self._is_fitted,
            'task': 'mono_classification',
            'description': 'Single-label genre classification (main genre only)',
            'model_name': save_path.stem
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"✅ Модель сохранена: {save_path}")
        print(f"✅ Метаданные: {meta_path}")

        return save_path

    def load(self, filepath: Optional[Path] = None, name: Optional[str] = None) -> 'XGBoostGenreClassifier':
        """
        Загружает модель и её конфигурацию

        Args:
            filepath: полный путь к файлу (если указан, приоритет выше)
            name: имя файла (загружается из директории по умолчанию)

        Returns:
            self

        Examples:
            # Загрузить с именем по умолчанию (xgboost_auto.json)
            model.load()

            # Загрузить с указанным именем
            model.load(name="my_best_model")

            # Загрузить по полному пути
            model.load(filepath=Path("./custom/path/model.json"))
        """
        # Определяем путь для загрузки
        if filepath is not None:
            # Используем полный путь
            load_path = Path(filepath)
        elif name is not None:
            # Используем имя в директории по умолчанию
            load_path = self._get_default_path(name)
        else:
            # Используем имя по умолчанию
            load_path = self._get_default_path()

        load_path = Path(load_path)

        # Проверяем существование
        if not load_path.exists():
            # Пробуем без расширения
            alt_path = load_path.with_suffix('')
            if alt_path.exists():
                load_path = alt_path
            else:
                # Ищем в директории по умолчанию
                models_dir = project_paths.xgboost_mono_models_dir
                if models_dir.exists():
                    candidates = list(models_dir.glob("*.json"))
                    if candidates:
                        load_path = candidates[0]
                        print(f"⚠️ Файл не найден, загружаем последний: {load_path.name}")
                    else:
                        raise FileNotFoundError(
                            f"Модель не найдена.\n"
                            f"Искали: {load_path}\n"
                            f"Директория: {models_dir}\n"
                            f"Доступные модели: {[f.name for f in models_dir.glob('*.json')] if models_dir.exists() else 'нет'}"
                        )
                else:
                    raise FileNotFoundError(f"Модель не найдена: {load_path}")

        # Загружаем модель
        self.sklearn_model = xgb.XGBClassifier()
        self.sklearn_model.load_model(str(load_path))
        self.model = self.sklearn_model.get_booster()

        # Загружаем метаданные
        meta_path = load_path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.params = meta.get('params', {})
            self.use_class_weights = meta.get('use_class_weights', True)
            self.genre_names = meta.get('genre_names')
            self.class_weights = meta.get('class_weights')
            # Сохраняем имя загруженной модели
            self.model_name = meta.get('model_name', load_path.stem)

        self._is_fitted = True
        print(f"✅ Модель загружена: {load_path}")

        return self

    def print_summary(self):
        """Выводит сводку о модели"""
        if not self._is_fitted:
            print("Модель не обучена")
            return

        print("=" * 60)
        print("СВОДКА МОДЕЛИ XGBOOST (MONO CLASSIFICATION)")
        print("=" * 60)
        print(f"Имя модели: {self.model_name}")
        print(f"Классов: {self.params.get('num_class', '?')}")
        print(f"Деревьев: {self.params.get('n_estimators', '?')}")
        print(f"Глубина: {self.params.get('max_depth', '?')}")
        print(f"Learning rate: {self.params.get('learning_rate', '?')}")
        print(f"Веса классов: {self.use_class_weights}")
        if self.class_weights:
            print("Веса классов:", self.class_weights)
        print("=" * 60)