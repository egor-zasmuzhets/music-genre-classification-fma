"""
src/data/load_processed.py
Быстрая загрузка уже подготовленных данных из кэша.
Используется после того, как DataPipeline уже запущен хотя бы раз.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.config import paths


class LoadProcessedData:
    """
    Быстрая загрузка подготовленных данных из data/processed/

    Использование:
        data = LoadProcessedData(subset="medium").load()
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']

    Или ещё короче:
        data = load_data()  # использует настройки из config
    """

    def __init__(
            self,
            subset: Optional[str] = None,
            min_samples_per_genre: Optional[int] = None,
            cache_dir: Optional[Path] = None
    ):
        """
        Args:
            subset: 'small', 'medium', 'large' (если None — из config)
            min_samples_per_genre: минимальное количество треков на жанр (если None — 100)
            cache_dir: директория с кэшем (по умолчанию из config)
        """
        # Берём значения из config, если не указаны
        self.subset = subset or paths.active_subset
        self.min_samples_per_genre = min_samples_per_genre or 100
        self.cache_dir = cache_dir or paths.processed_data_dir

        self._data = None

    def _get_cache_file(self) -> Path:
        """Возвращает путь к файлу кэша для текущих параметров"""
        return self.cache_dir / f'pipeline_{self.subset}_min{self.min_samples_per_genre}.pkl'

    def _get_metadata_path(self) -> Path:
        """Возвращает путь к файлу метаданных"""
        return self.cache_dir / 'metadata.json'

    def exists(self) -> bool:
        """Проверяет, есть ли подготовленные данные в кэше"""
        return self._get_cache_file().exists() and self._get_metadata_path().exists()

    def load(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Загружает подготовленные данные из кэша

        Args:
            force_reload: игнорировать кэш и загрузить заново из файлов .npy

        Returns:
            Словарь с ключами:
            - X_train, X_val, X_test
            - y_train, y_val, y_test
            - label_encoder, scaler
            - genre_names
            - metadata
        """
        if not self.exists():
            raise FileNotFoundError(
                f"Данные не найдены в {self.cache_dir}\n"
                f"Сначала запустите DataPipeline для FMA {self.subset}"
            )

        if self._data is not None and not force_reload:
            return self._data

        print(f"Загрузка данных из {self.cache_dir}...")

        # Загружаем массивы
        X_train = np.load(self.cache_dir / 'X_train.npy')
        X_val = np.load(self.cache_dir / 'X_val.npy')
        X_test = np.load(self.cache_dir / 'X_test.npy')
        y_train = np.load(self.cache_dir / 'y_train.npy')
        y_val = np.load(self.cache_dir / 'y_val.npy')
        y_test = np.load(self.cache_dir / 'y_test.npy')

        train_indices_path = self.cache_dir / 'train_indices.npy'
        if train_indices_path.exists():
            train_indices = np.load(train_indices_path)
            val_indices = np.load(self.cache_dir / 'val_indices.npy')
            test_indices = np.load(self.cache_dir / 'test_indices.npy')
        else:
            train_indices = val_indices = test_indices = None

        # Загружаем метаданные
        with open(self.cache_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Загружаем препроцессоры
        import joblib
        preprocessor_path = self.cache_dir / 'preprocessor.pkl'
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            label_encoder = preprocessor['label_encoder']
            scaler = preprocessor['scaler']
        else:
            # Если нет единого файла, пробуем отдельные
            label_encoder = None
            scaler = None

        self._data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'label_encoder': label_encoder,
            'scaler': scaler,
            'genre_names': metadata.get('class_names', []),
            'metadata': metadata
        }

        print(f"✅ Загружены данные: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
        print(f"   Жанров: {metadata.get('num_classes', '?')}")

        return self._data

    @property
    def load_to_dataframe(self) -> Dict[str, Any]:
        """
        Загружает данные и преобразует обратно в DataFrame (если нужно)
        Для XGBoost удобнее numpy, но иногда нужен DataFrame
        """
        data = self.load()

        # Восстанавливаем имена признаков из метаданных
        feature_names = data['metadata'].get('feature_names')

        if feature_names:
            X_train = pd.DataFrame(data['X_train'], columns=feature_names)
            X_val = pd.DataFrame(data['X_val'], columns=feature_names)
            X_test = pd.DataFrame(data['X_test'], columns=feature_names)
        else:
            X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': data['y_train'],
            'y_val': data['y_val'],
            'y_test': data['y_test'],
            'label_encoder': data['label_encoder'],
            'scaler': data['scaler'],
            'genre_names': data['genre_names'],
            'metadata': data['metadata']
        }

    def print_info(self):
        """Выводит информацию о загруженных данных"""
        if not self.exists():
            print(f"❌ Данные не найдены в {self.cache_dir}")
            return

        with open(self.cache_dir / 'metadata.json', 'r') as f:
            meta = json.load(f)

        print("=" * 50)
        print("ЗАГРУЖЕННЫЕ ДАННЫЕ")
        print("=" * 50)
        print(f"Датасет:        FMA {meta.get('subset', '?').upper()}")
        print(f"Треков:         {meta.get('num_samples', '?')}")
        print(f"Признаков:      {meta.get('num_features', '?')}")
        print(f"Жанров:         {meta.get('num_classes', '?')}")
        print(f"\nРазбиение:")
        print(f"  Train:        {meta.get('train_size', '?')}")
        print(f"  Val:          {meta.get('val_size', '?')}")
        print(f"  Test:         {meta.get('test_size', '?')}")
        print(f"\nЖанры: {', '.join(meta.get('class_names', [])[:5])}...")


def load_track_indices(
        subset: Optional[str] = None,
        min_samples_per_genre: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Быстрая загрузка индексов треков для train/val/test

    Returns:
        (train_indices, val_indices, test_indices)
    """
    loader = LoadProcessedData(subset=subset, min_samples_per_genre=min_samples_per_genre)
    data = loader.load()
    return data['train_indices'], data['val_indices'], data['test_indices']


def load_data(
        subset: Optional[str] = None,
        min_samples_per_genre: int = 100,
        as_dataframe: bool = False
) -> Dict[str, Any]:
    """
    Быстрая загрузка подготовленных данных (однострочник)

    Пример:
        data = load_data()
        X_train = data['X_train']

    Args:
        subset: 'small', 'medium', 'large' (из config если None)
        min_samples_per_genre: минимальное количество треков на жанр
        as_dataframe: вернуть DataFrame вместо numpy (для анализа)

    Returns:
        Словарь с данными
    """
    loader = LoadProcessedData(subset=subset, min_samples_per_genre=min_samples_per_genre)
    if as_dataframe:
        return loader.load_to_dataframe
    return loader.load()


# Короткий alias
load = load_data