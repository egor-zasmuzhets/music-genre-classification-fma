"""
tests/data/test_preprocessor.py
Тесты для DataPreprocessor на реальных данных FMA Small
"""

import pytest
import numpy as np
import pandas as pd
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Тесты препроцессора"""

    def test_preprocessor_initialization(self):
        """Проверка инициализации"""
        preprocessor = DataPreprocessor(min_samples_per_genre=50)
        assert preprocessor.min_samples_per_genre == 50
        assert not preprocessor._is_fitted

    def test_filter_rare_genres(self, fma_small_tracks, genre_col):
        """Проверка фильтрации редких жанров"""
        preprocessor = DataPreprocessor(min_samples_per_genre=50)

        # Подсчитываем жанры до фильтрации
        original_genres = fma_small_tracks[genre_col].value_counts()
        original_count = len(original_genres)

        # Фильтруем
        filtered = preprocessor.filter_rare_genres(fma_small_tracks, genre_col)

        # Проверяем, что количество жанров уменьшилось или осталось тем же
        filtered_genres = filtered[genre_col].value_counts()
        assert len(filtered_genres) <= original_count

        # Проверяем, что все оставшиеся жанры имеют >= min_samples
        for genre, count in filtered_genres.items():
            assert count >= preprocessor.min_samples_per_genre, \
                f"Жанр {genre} имеет {count} треков, что меньше {preprocessor.min_samples_per_genre}"

    def test_encode_labels(self, fma_small_tracks, genre_col):
        """Проверка кодирования меток"""
        preprocessor = DataPreprocessor(min_samples_per_genre=50)

        # Подготавливаем данные
        filtered = preprocessor.filter_rare_genres(fma_small_tracks, genre_col)

        # Разделяем на train/val/test (берём часть данных)
        y_all = filtered[genre_col]
        n = len(y_all)
        y_train = y_all[:int(n * 0.8)]
        y_val = y_all[int(n * 0.8):int(n * 0.9)]
        y_test = y_all[int(n * 0.9):]

        # Кодируем
        y_train_enc, y_val_enc, y_test_enc = preprocessor.encode_labels(
            y_train, y_val, y_test
        )

        # Проверяем типы и размеры
        assert isinstance(y_train_enc, np.ndarray)
        assert len(y_train_enc) == len(y_train)

        if y_val_enc is not None:
            assert len(y_val_enc) == len(y_val)

        if y_test_enc is not None:
            assert len(y_test_enc) == len(y_test)

        # Проверяем, что метки — целые числа от 0 до num_classes-1
        assert y_train_enc.min() >= 0
        assert y_train_enc.max() < len(preprocessor.label_encoder.classes_)

    def test_normalize_features(self, fma_small_features):
        """Проверка нормализации признаков"""
        preprocessor = DataPreprocessor(min_samples_per_genre=50)

        X_all = fma_small_features

        # Разделяем
        n = len(X_all)
        X_train = X_all[:int(n * 0.8)]
        X_val = X_all[int(n * 0.8):int(n * 0.9)]
        X_test = X_all[int(n * 0.9):]

        # Нормализуем
        X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.normalize_features(
            X_train, X_val, X_test
        )

        # Проверяем, что после нормализации среднее ~0, std ~1
        assert abs(X_train_scaled.mean()) < 1e-6, \
            f"Среднее после нормализации: {X_train_scaled.mean()}"
        assert abs(X_train_scaled.std() - 1.0) < 1e-5, \
            f"Стандартное отклонение после нормализации: {X_train_scaled.std()}"

        # Проверяем размеры
        assert X_train_scaled.shape[0] == len(X_train)
        assert X_train_scaled.shape[1] == X_train.shape[1]

        if X_val_scaled is not None:
            assert X_val_scaled.shape[1] == X_train.shape[1]

        if X_test_scaled is not None:
            assert X_test_scaled.shape[1] == X_train.shape[1]

    def test_get_class_weights(self, fma_small_tracks, genre_col):
        """Проверка вычисления весов классов"""
        preprocessor = DataPreprocessor(min_samples_per_genre=50)

        # Подготавливаем данные
        filtered = preprocessor.filter_rare_genres(fma_small_tracks, genre_col)
        y_all = filtered[genre_col]

        # Кодируем метки (только train)
        y_train_enc, _, _ = preprocessor.encode_labels(y_all[:int(len(y_all) * 0.8)], None, None)

        # Вычисляем веса
        class_weights = preprocessor.get_class_weights(y_train_enc)

        # Проверяем
        assert isinstance(class_weights, dict)
        # Количество весов должно равняться количеству уникальных классов
        assert len(class_weights) == len(np.unique(y_train_enc))

        # Веса должны быть положительными
        assert all(w > 0 for w in class_weights.values())

    def test_save_load_preprocessor(self, tmp_path, fma_small_tracks, genre_col):
        """Проверка сохранения и загрузки препроцессора"""
        # Создаём и обучаем препроцессор
        preprocessor_original = DataPreprocessor(min_samples_per_genre=50)

        filtered = preprocessor_original.filter_rare_genres(fma_small_tracks, genre_col)
        y_all = filtered[genre_col]
        preprocessor_original.encode_labels(y_all, None, None)

        # Берём признаки для нормализации
        from src.data.loader import FMALoader
        features = FMALoader().features
        common_idx = filtered.index.intersection(features.index)
        X = features.loc[common_idx].iloc[:100]
        preprocessor_original.normalize_features(X, None, None)

        # Сохраняем
        save_path = tmp_path / "test_preprocessor.pkl"
        preprocessor_original.save(save_path)
        assert save_path.exists()

        # Загружаем в новый объект
        preprocessor_loaded = DataPreprocessor()
        preprocessor_loaded.load(save_path)

        # Проверяем, что параметры совпадают
        assert preprocessor_loaded.min_samples_per_genre == preprocessor_original.min_samples_per_genre
        assert preprocessor_loaded._is_fitted == preprocessor_original._is_fitted

        # Проверяем, что label_encoder одинаковый
        assert list(preprocessor_loaded.label_encoder.classes_) == \
               list(preprocessor_original.label_encoder.classes_)