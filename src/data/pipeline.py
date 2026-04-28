"""
src/data/pipeline.py
Полный пайплайн подготовки данных для классификации
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pickle
import joblib

from src.data.config import paths
from src.data.loader import FMALoader
from src.data.preprocessor import DataPreprocessor


class DataPipeline:
    """
    Полный пайплайн подготовки данных:

    1. Загрузка метаданных (tracks, features, genres)
    2. Фильтрация по подмножеству (small/medium/large)
    3. Фильтрация по минимальному количеству треков на жанр
    4. Использование официального разбиения FMA (train/val/test)
    5. Сопоставление признаков и меток
    6. Кодирование меток и нормализация признаков
    7. Сохранение подготовленных данных
    """

    def __init__(
            self,
            subset: str = "medium",
            min_samples_per_genre: int = 100,
            use_features: bool = True,
            cache_dir: Optional[Path] = None
    ):
        """
        Args:
            subset: 'small', 'medium', или 'large'
            min_samples_per_genre: минимальное количество треков для жанра
            use_features: использовать ли предвычисленные признаки (features.csv)
            cache_dir: директория для кэширования (по умолчанию из config)
        """
        self.subset = subset
        self.min_samples_per_genre = min_samples_per_genre
        self.use_features = use_features
        self.cache_dir = cache_dir or paths.processed_data_dir

        # Инициализация компонентов
        self.loader = FMALoader()
        self.preprocessor = DataPreprocessor(min_samples_per_genre)

        # Кэш для результатов
        self._data = None

    def run(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Запускает полный пайплайн подготовки данных

        Args:
            force_reload: принудительно перезапустить пайплайн (игнорировать кэш)

        Returns:
            Словарь с подготовленными данными:
            - X_train, X_val, X_test: признаки
            - y_train, y_val, y_test: метки
            - label_encoder, scaler: препроцессоры
            - genre_names: список жанров
            - metadata: информация о датасете
        """
        # Проверяем кэш
        if not force_reload and self._is_cached():
            print("Загрузка данных из кэша...")
            return self._load_from_cache()

        print("=" * 60)
        print(f"Запуск пайплайна подготовки данных (FMA {self.subset})")
        print("=" * 60)

        # Шаг 1: Загрузка треков и фильтрация по подмножеству
        tracks = self.loader.get_tracks_by_subset(self.subset)

        # Шаг 2: Удаление треков без жанра
        genre_col = ('track', 'genre_top')
        if genre_col not in tracks.columns:
            raise KeyError(f"Колонка {genre_col} не найдена")

        tracks_with_genre = tracks[tracks[genre_col].notna()].copy()
        print(f"Треков с жанром: {len(tracks_with_genre)}")

        # Шаг 3: Фильтрация редких жанров
        tracks_filtered = self.preprocessor.filter_rare_genres(
            tracks_with_genre, genre_col
        )

        # Шаг 4: Получение официального разбиения
        splits = self.loader.get_available_splits(tracks_filtered)

        train_idx = splits['training']
        val_idx = splits['validation']
        test_idx = splits['test']

        print(f"\nОфициальное разбиение FMA:")
        print(f"  Train: {len(train_idx)} ({len(train_idx) / len(tracks_filtered) * 100:.1f}%)")
        print(f"  Val:   {len(val_idx)} ({len(val_idx) / len(tracks_filtered) * 100:.1f}%)")
        print(f"  Test:  {len(test_idx)} ({len(test_idx) / len(tracks_filtered) * 100:.1f}%)")

        # Шаг 5: Получение признаков
        if self.use_features:
            features_all = self.loader.features

            # Общие индексы между треками и признаками
            common_idx = tracks_filtered.index.intersection(features_all.index)
            if len(common_idx) != len(tracks_filtered):
                print(f"Предупреждение: {len(tracks_filtered) - len(common_idx)} треков без признаков")

            X_all = features_all.loc[common_idx]
            y_all = tracks_filtered.loc[common_idx, genre_col]

            # Обновляем индексы для выборок
            train_idx = common_idx.intersection(train_idx)
            val_idx = common_idx.intersection(val_idx)
            test_idx = common_idx.intersection(test_idx)
        else:
            # Если не используем признаки, берем только метаданные (например, bit_rate)
            X_all = tracks_filtered[['track', 'bit_rate']].copy()
            y_all = tracks_filtered[genre_col]

        # Шаг 6: Разделение на выборки
        X_train_raw = X_all.loc[train_idx]
        X_val_raw = X_all.loc[val_idx]
        X_test_raw = X_all.loc[test_idx]

        y_train_raw = y_all.loc[train_idx]
        y_val_raw = y_all.loc[val_idx]
        y_test_raw = y_all.loc[test_idx]

        print(f"\nРазмеры выборок после сопоставления:")
        print(f"  X_train: {X_train_raw.shape}")
        print(f"  X_val:   {X_val_raw.shape}")
        print(f"  X_test:  {X_test_raw.shape}")

        # Шаг 7: Кодирование меток
        y_train, y_val, y_test = self.preprocessor.encode_labels(
            y_train_raw, y_val_raw, y_test_raw
        )

        # Шаг 8: Нормализация признаков
        X_train, X_val, X_test = self.preprocessor.normalize_features(
            X_train_raw, X_val_raw, X_test_raw
        )

        # Шаг 9: Сбор метаданных
        metadata = {
            'subset': self.subset,
            'num_samples': len(X_all),
            'num_features': X_all.shape[1],
            'num_classes': len(self.preprocessor.label_encoder.classes_),
            'class_names': list(self.preprocessor.label_encoder.classes_),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'min_samples_per_genre': self.min_samples_per_genre,
            'use_features': self.use_features,
            'split_method': 'official_fma_80_10_10',
            'feature_names': list(X_all.columns) if isinstance(X_all, pd.DataFrame) else None
        }

        # Шаг 10: Формирование результата
        self._data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': self.preprocessor.label_encoder,
            'scaler': self.preprocessor.scaler,
            'genre_names': metadata['class_names'],
            'metadata': metadata
        }

        # Шаг 11: Сохранение в кэш
        self._save_to_cache()

        return self._data

    def _is_cached(self) -> bool:
        """Проверяет, есть ли кэшированные данные"""
        cache_file = self.cache_dir / f'pipeline_{self.subset}_min{self.min_samples_per_genre}.pkl'
        return cache_file.exists()

    def _save_to_cache(self):
        """Сохраняет подготовленные данные в кэш"""
        if self._data is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем основной объект
        cache_file = self.cache_dir / f'pipeline_{self.subset}_min{self.min_samples_per_genre}.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(self._data, f)

        # Также сохраняем отдельные файлы для удобства использования
        np.save(self.cache_dir / 'X_train.npy', self._data['X_train'])
        np.save(self.cache_dir / 'X_val.npy', self._data['X_val'])
        np.save(self.cache_dir / 'X_test.npy', self._data['X_test'])
        np.save(self.cache_dir / 'y_train.npy', self._data['y_train'])
        np.save(self.cache_dir / 'y_val.npy', self._data['y_val'])
        np.save(self.cache_dir / 'y_test.npy', self._data['y_test'])

        # Сохраняем препроцессоры
        self.preprocessor.save(self.cache_dir / 'preprocessor.pkl')

        # Сохраняем метаданные
        with open(self.cache_dir / 'metadata.json', 'w') as f:
            import json
            # Преобразуем numpy типы
            metadata_copy = self._data['metadata'].copy()
            metadata_copy['class_names'] = list(metadata_copy['class_names'])
            json.dump(metadata_copy, f, indent=2)

        print(f"\n✅ Данные сохранены в {self.cache_dir}")

    def _load_from_cache(self) -> Dict[str, Any]:
        """Загружает данные из кэша"""
        cache_file = self.cache_dir / f'pipeline_{self.subset}_min{self.min_samples_per_genre}.pkl'

        with open(cache_file, 'rb') as f:
            self._data = pickle.load(f)

        # Загружаем препроцессор
        self.preprocessor.load(self.cache_dir / 'preprocessor.pkl')

        print(f"✅ Данные загружены из кэша")
        return self._data

    def get_class_weights(self) -> Dict[int, float]:
        """Возвращает веса классов для борьбы с дисбалансом"""
        if self._data is None:
            raise ValueError("Сначала запустите pipeline.run()")
        return self.preprocessor.get_class_weights(self._data['y_train'])

    def print_summary(self):
        """Выводит сводку о подготовленных данных"""
        if self._data is None:
            print("Данные не подготовлены. Запустите pipeline.run()")
            return

        meta = self._data['metadata']
        print("=" * 60)
        print("СВОДКА ПОДГОТОВЛЕННЫХ ДАННЫХ")
        print("=" * 60)
        print(f"Датасет:     FMA {meta['subset'].upper()}")
        print(f"Треков:      {meta['num_samples']}")
        print(f"Признаков:   {meta['num_features']}")
        print(f"Жанров:      {meta['num_classes']}")
        print(f"\nРазделение (официальное FMA):")
        print(f"  Train:     {meta['train_size']} ({meta['train_size'] / meta['num_samples'] * 100:.1f}%)")
        print(f"  Val:       {meta['val_size']} ({meta['val_size'] / meta['num_samples'] * 100:.1f}%)")
        print(f"  Test:      {meta['test_size']} ({meta['test_size'] / meta['num_samples'] * 100:.1f}%)")
        print(f"\nЖанры:")
        for i, genre in enumerate(meta['class_names']):
            print(f"  {i}: {genre}")