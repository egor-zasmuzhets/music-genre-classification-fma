"""
src/data/preprocessor.py
Предобработка данных: кодирование меток, нормализация, фильтрация
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path


class DataPreprocessor:
    """
    Предобработка данных для классификации жанров.

    Выполняет:
    - Фильтрацию редких жанров
    - Кодирование меток (LabelEncoder)
    - Нормализацию признаков (StandardScaler)
    - Вычисление весов классов
    """

    def __init__(self, min_samples_per_genre: int = 100):
        """
        Args:
            min_samples_per_genre: минимальное количество треков для жанра
        """
        self.min_samples_per_genre = min_samples_per_genre
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._is_fitted = False

    def filter_rare_genres(
            self,
            tracks_df: pd.DataFrame,
            genre_col: Tuple[str, str]
    ) -> pd.DataFrame:
        """
        Удаляет жанры с количеством треков меньше min_samples_per_genre

        Args:
            tracks_df: DataFrame с треками
            genre_col: колонка с жанрами (например, ('track', 'genre_top'))

        Returns:
            Отфильтрованный DataFrame
        """
        genre_counts = tracks_df[genre_col].value_counts()
        rare_genres = genre_counts[genre_counts < self.min_samples_per_genre].index
        common_genres = genre_counts[genre_counts >= self.min_samples_per_genre].index

        if len(rare_genres) > 0:
            print(f"Удалено редких жанров: {len(rare_genres)}")
            for g in rare_genres:
                print(f"  - {g}: {genre_counts[g]} треков")

        filtered = tracks_df[tracks_df[genre_col].isin(common_genres)].copy()
        print(f"Осталось жанров: {len(common_genres)}")
        print(f"Осталось треков: {len(filtered)}")

        return filtered

    def encode_labels(
            self,
            y_train: pd.Series,
            y_val: Optional[pd.Series] = None,
            y_test: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Кодирует метки (fit на train, transform на val/test)

        Args:
            y_train: метки тренировочной выборки
            y_val: метки валидационной выборки (опционально)
            y_test: метки тестовой выборки (опционально)

        Returns:
            (y_train_encoded, y_val_encoded, y_test_encoded)
        """
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        y_val_encoded = None
        if y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)

        y_test_encoded = None
        if y_test is not None:
            y_test_encoded = self.label_encoder.transform(y_test)

        print(f"Закодировано классов: {len(self.label_encoder.classes_)}")
        for i, name in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {name}")

        return y_train_encoded, y_val_encoded, y_test_encoded

    def normalize_features(
            self,
            X_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Нормализует признаки (fit на train, transform на val/test)

        Args:
            X_train: признаки тренировочной выборки
            X_val: признаки валидационной выборки (опционально)
            X_test: признаки тестовой выборки (опционально)

        Returns:
            (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"X_train: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")

        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            print(f"X_val:   mean={X_val_scaled.mean():.4f}, std={X_val_scaled.std():.4f}")

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            print(f"X_test:  mean={X_test_scaled.mean():.4f}, std={X_test_scaled.std():.4f}")

        self._is_fitted = True
        return X_train_scaled, X_val_scaled, X_test_scaled

    def get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Вычисляет веса классов для борьбы с дисбалансом

        Args:
            y_train: закодированные метки тренировочной выборки

        Returns:
            Словарь {class_id: weight}
        """
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        return dict(zip(classes, weights))

    def save(self, path: Path):
        """Сохраняет препроцессор"""
        joblib.dump({
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'min_samples_per_genre': self.min_samples_per_genre
        }, path)
        print(f"Препроцессор сохранён: {path}")

    def load(self, path: Path):
        """Загружает препроцессор"""
        data = joblib.load(path)
        self.label_encoder = data['label_encoder']
        self.scaler = data['scaler']
        self.min_samples_per_genre = data['min_samples_per_genre']
        self._is_fitted = True
        print(f"Препроцессор загружен: {path}")