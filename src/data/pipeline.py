"""
src/data/pipeline.py
Полный пайплайн подготовки данных для классификации

Поддерживает:
- Загрузку метаданных и признаков из CSV
- Фильтрацию по подмножеству и редким жанрам
- Официальное разбиение FMA (train/val/test)
- Кодирование меток и нормализацию признаков
- Сохранение индексов треков для последующей загрузки аудио
- Проверку доступности аудио в ZIP архиве
- Детальные отчёты о пропущенных треках
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import pickle
import json
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
    7. Сохранение подготовленных данных и индексов треков
    8. Проверка доступности аудио в ZIP архиве
    """

    def __init__(
            self,
            subset: str = "medium",
            min_samples_per_genre: int = 100,
            use_features: bool = True,
            cache_dir: Optional[Path] = None,
            check_audio_availability: bool = True
    ):
        """
        Args:
            subset: 'small', 'medium', или 'large'
            min_samples_per_genre: минимальное количество треков для жанра
            use_features: использовать ли предвычисленные признаки (features.csv)
            cache_dir: директория для кэширования (по умолчанию из config)
            check_audio_availability: проверять ли доступность аудио в ZIP
        """
        self.subset = subset
        self.min_samples_per_genre = min_samples_per_genre
        self.use_features = use_features
        self.cache_dir = cache_dir or paths.processed_data_dir
        self.check_audio_availability = check_audio_availability

        # Инициализация компонентов
        self.loader = FMALoader()
        self.preprocessor = DataPreprocessor(min_samples_per_genre)

        # Кэш для результатов
        self._data = None

        # Отслеживание проблемных треков
        self._missing_audio_report = None

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
            - train_indices, val_indices, test_indices: индексы треков
            - audio_availability: информация о доступности аудио
        """
        # Проверяем кэш
        if not force_reload and self._is_cached():
            print("Загрузка данных из кэша...")
            return self._load_from_cache()

        print("=" * 60)
        print(f"🚀 ЗАПУСК ПАЙПЛАЙНА ПОДГОТОВКИ ДАННЫХ (FMA {self.subset.upper()})")
        print("=" * 60)

        # Шаг 1: Загрузка треков и фильтрация по подмножеству
        tracks = self.loader.get_tracks_by_subset(self.subset)

        # Шаг 2: Удаление треков без жанра
        genre_col = ('track', 'genre_top')
        if genre_col not in tracks.columns:
            raise KeyError(f"Колонка {genre_col} не найдена")

        tracks_with_genre = tracks[tracks[genre_col].notna()].copy()
        print(f"\n📊 Шаг 1/9: Загружено треков с жанром: {len(tracks_with_genre)}")

        # Шаг 3: Фильтрация редких жанров
        tracks_filtered = self.preprocessor.filter_rare_genres(
            tracks_with_genre, genre_col
        )

        # Шаг 4: Получение официального разбиения
        splits = self.loader.get_available_splits(tracks_filtered)

        train_idx = splits['training']
        val_idx = splits['validation']
        test_idx = splits['test']

        print(f"\n📊 Шаг 2/9: Официальное разбиение FMA:")
        print(f"   Train: {len(train_idx)} ({len(train_idx) / len(tracks_filtered) * 100:.1f}%)")
        print(f"   Val:   {len(val_idx)} ({len(val_idx) / len(tracks_filtered) * 100:.1f}%)")
        print(f"   Test:  {len(test_idx)} ({len(test_idx) / len(tracks_filtered) * 100:.1f}%)")

        # Шаг 5: Получение признаков
        if self.use_features:
            features_all = self.loader.features

            # Общие индексы между треками и признаками
            common_idx = tracks_filtered.index.intersection(features_all.index)
            if len(common_idx) != len(tracks_filtered):
                missing_features = len(tracks_filtered) - len(common_idx)
                print(f"   ⚠️ Предупреждение: {missing_features} треков без признаков в features.csv")

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

        print(f"\n📊 Шаг 3/9: Размеры выборок после сопоставления:")
        print(f"   X_train: {X_train_raw.shape}")
        print(f"   X_val:   {X_val_raw.shape}")
        print(f"   X_test:  {X_test_raw.shape}")

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
            'metadata': metadata,
            'train_indices': train_idx,
            'val_indices': val_idx,
            'test_indices': test_idx,
            'missing_audio_report': None
        }

        # Шаг 11: Проверка доступности аудио в ZIP
        if self.check_audio_availability:
            self._check_audio_availability(train_idx, val_idx, test_idx)

        # Шаг 12: Сохранение в кэш
        self._save_to_cache()

        # Финальная сводка
        self._print_final_summary()

        return self._data

    def _check_audio_availability(
        self,
        train_idx: pd.Index,
        val_idx: pd.Index,
        test_idx: pd.Index
    ):
        """
        Проверяет доступность аудио в ZIP архиве

        Args:
            train_idx: индексы тренировочной выборки
            val_idx: индексы валидационной выборки
            test_idx: индексы тестовой выборки
        """
        print("\n📊 Шаг 4/9: Проверка доступности аудио в ZIP архиве...")

        from src.data.audio_loader import AudioLoader

        try:
            audio_loader = AudioLoader(use_disk_cache=False)
            available_tracks = set(audio_loader.get_available_tracks())

            # Функция для анализа доступности
            def analyze_availability(indices, name):
                indices_list = list(indices)
                available = [idx for idx in indices_list if idx in available_tracks]
                missing = [idx for idx in indices_list if idx not in available_tracks]
                return {
                    'total': len(indices_list),
                    'available': len(available),
                    'missing': len(missing),
                    'missing_list': missing,
                    'available_rate': len(available) / len(indices_list) if indices_list else 0
                }

            train_analysis = analyze_availability(train_idx, 'train')
            val_analysis = analyze_availability(val_idx, 'val')
            test_analysis = analyze_availability(test_idx, 'test')

            self._data['audio_availability'] = {
                'train': train_analysis,
                'val': val_analysis,
                'test': test_analysis
            }

            # Формируем отчёт о пропущенных треках
            self._missing_audio_report = {
                'train_missing': train_analysis['missing_list'],
                'val_missing': val_analysis['missing_list'],
                'test_missing': test_analysis['missing_list'],
                'total_missing': train_analysis['missing'] + val_analysis['missing'] + test_analysis['missing']
            }

            # Выводим предупреждения
            if train_analysis['missing'] > 0:
                print(f"   ⚠️ Train: {train_analysis['missing']}/{train_analysis['total']} треков отсутствуют")
            if val_analysis['missing'] > 0:
                print(f"   ⚠️ Val:   {val_analysis['missing']}/{val_analysis['total']} треков отсутствуют")
            if test_analysis['missing'] > 0:
                print(f"   ⚠️ Test:  {test_analysis['missing']}/{test_analysis['total']} треков отсутствуют")

            if self._missing_audio_report['total_missing'] > 0:
                print(f"\n   📋 Детальный отчёт о пропущенных треках сохранён в self._data['missing_audio_report']")

            audio_loader.close()

        except Exception as e:
            print(f"   ⚠️ Не удалось проверить доступность аудио: {e}")
            self._data['audio_availability'] = None

    def _print_final_summary(self):
        """Выводит финальную сводку по пайплайну"""
        print("\n" + "=" * 60)
        print("📊 ФИНАЛЬНАЯ СВОДКА ПАЙПЛАЙНА")
        print("=" * 60)

        meta = self._data['metadata']
        print(f"Датасет:     FMA {meta['subset'].upper()}")
        print(f"Треков:      {meta['num_samples']}")
        print(f"Признаков:   {meta['num_features']}")
        print(f"Жанров:      {meta['num_classes']}")

        print(f"\nРазделение (официальное FMA):")
        print(f"  Train:     {meta['train_size']} ({meta['train_size'] / meta['num_samples'] * 100:.1f}%)")
        print(f"  Val:       {meta['val_size']} ({meta['val_size'] / meta['num_samples'] * 100:.1f}%)")
        print(f"  Test:      {meta['test_size']} ({meta['test_size'] / meta['num_samples'] * 100:.1f}%)")

        if self._data.get('audio_availability'):
            audio = self._data['audio_availability']
            print(f"\nДоступность аудио в ZIP:")
            print(f"  Train: {audio['train']['available']}/{audio['train']['total']} ({audio['train']['available_rate']:.1%})")
            print(f"  Val:   {audio['val']['available']}/{audio['val']['total']} ({audio['val']['available_rate']:.1%})")
            print(f"  Test:  {audio['test']['available']}/{audio['test']['total']} ({audio['test']['available_rate']:.1%})")

        print("=" * 60)

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

        # Сохраняем отдельные файлы для удобства
        np.save(self.cache_dir / 'X_train.npy', self._data['X_train'])
        np.save(self.cache_dir / 'X_val.npy', self._data['X_val'])
        np.save(self.cache_dir / 'X_test.npy', self._data['X_test'])
        np.save(self.cache_dir / 'y_train.npy', self._data['y_train'])
        np.save(self.cache_dir / 'y_val.npy', self._data['y_val'])
        np.save(self.cache_dir / 'y_test.npy', self._data['y_test'])

        # Сохраняем индексы треков
        train_indices = self._data.get('train_indices')
        val_indices = self._data.get('val_indices')
        test_indices = self._data.get('test_indices')

        if train_indices is not None:
            np.save(self.cache_dir / 'train_indices.npy', train_indices)
            np.save(self.cache_dir / 'val_indices.npy', val_indices)
            np.save(self.cache_dir / 'test_indices.npy', test_indices)

        # Сохраняем отчёт о пропущенных треках
        if self._missing_audio_report:
            with open(self.cache_dir / 'missing_audio_report.json', 'w') as f:
                # Конвертируем numpy типы в Python типы
                report_copy = {
                    'train_missing': [int(x) for x in self._missing_audio_report['train_missing']],
                    'val_missing': [int(x) for x in self._missing_audio_report['val_missing']],
                    'test_missing': [int(x) for x in self._missing_audio_report['test_missing']],
                    'total_missing': self._missing_audio_report['total_missing']
                }
                json.dump(report_copy, f, indent=2)

        # Сохраняем препроцессоры
        self.preprocessor.save(self.cache_dir / 'preprocessor.pkl')

        # Сохраняем метаданные
        with open(self.cache_dir / 'metadata.json', 'w') as f:
            metadata_copy = self._data['metadata'].copy()
            metadata_copy['class_names'] = list(metadata_copy['class_names'])
            # Конвертируем numpy типы в Python типы
            if 'feature_names' in metadata_copy and metadata_copy['feature_names'] is not None:
                metadata_copy['feature_names'] = list(metadata_copy['feature_names'])
            json.dump(metadata_copy, f, indent=2)

        print(f"\n✅ Данные сохранены в {self.cache_dir}")

    def _load_from_cache(self) -> Dict[str, Any]:
        """Загружает данные из кэша"""
        cache_file = self.cache_dir / f'pipeline_{self.subset}_min{self.min_samples_per_genre}.pkl'

        with open(cache_file, 'rb') as f:
            self._data = pickle.load(f)

        # Загружаем индексы, если они есть
        train_indices_path = self.cache_dir / 'train_indices.npy'
        if train_indices_path.exists():
            self._data['train_indices'] = np.load(train_indices_path)
            self._data['val_indices'] = np.load(self.cache_dir / 'val_indices.npy')
            self._data['test_indices'] = np.load(self.cache_dir / 'test_indices.npy')

        # Загружаем препроцессор
        self.preprocessor.load(self.cache_dir / 'preprocessor.pkl')

        # Загружаем отчёт о пропущенных треках (если есть)
        missing_report_path = self.cache_dir / 'missing_audio_report.json'
        if missing_report_path.exists():
            with open(missing_report_path, 'r') as f:
                self._missing_audio_report = json.load(f)
                self._data['missing_audio_report'] = self._missing_audio_report

        print(f"\n✅ Данные загружены из кэша")

        # Выводим информацию о пропущенных треках из кэша
        if self._missing_audio_report and self._missing_audio_report.get('total_missing', 0) > 0:
            print(f"   ⚠️ В кэше: {self._missing_audio_report['total_missing']} треков отсутствуют в ZIP")

        return self._data

    def get_class_weights(self) -> Dict[int, float]:
        """Возвращает веса классов для борьбы с дисбалансом"""
        if self._data is None:
            raise ValueError("Сначала запустите pipeline.run()")
        return self.preprocessor.get_class_weights(self._data['y_train'])

    def get_missing_audio_report(self) -> Optional[Dict[str, Any]]:
        """
        Возвращает отчёт о треках, отсутствующих в ZIP архиве

        Returns:
            Словарь с информацией о пропущенных треках или None
        """
        return self._missing_audio_report

    def get_available_indices(self, split: str = 'train') -> List[int]:
        """
        Возвращает индексы треков, доступных в ZIP архиве для указанного сплита

        Args:
            split: 'train', 'val', или 'test'

        Returns:
            Список доступных индексов
        """
        if self._data is None:
            raise ValueError("Сначала запустите pipeline.run()")

        audio_avail = self._data.get('audio_availability')
        if audio_avail is None:
            raise ValueError("Информация о доступности аудио недоступна. Запустите pipeline с check_audio_availability=True")

        split_map = {
            'train': audio_avail['train']['available'],
            'val': audio_avail['val']['available'],
            'test': audio_avail['test']['available']
        }

        return split_map.get(split, [])

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

        # Выводим информацию о доступности аудио
        audio_avail = self._data.get('audio_availability')
        if audio_avail:
            print(f"\nДоступность аудио в ZIP:")
            print(f"  Train: {audio_avail['train']['available']}/{audio_avail['train']['total']} ({audio_avail['train']['available_rate']:.1%})")
            print(f"  Val:   {audio_avail['val']['available']}/{audio_avail['val']['total']} ({audio_avail['val']['available_rate']:.1%})")
            print(f"  Test:  {audio_avail['test']['available']}/{audio_avail['test']['total']} ({audio_avail['test']['available_rate']:.1%})")

            if self._missing_audio_report and self._missing_audio_report.get('total_missing', 0) > 0:
                print(f"\n⚠️ Внимание: {self._missing_audio_report['total_missing']} треков отсутствуют в ZIP архиве")

        print(f"\nЖанры:")
        for i, genre in enumerate(meta['class_names']):
            print(f"  {i}: {genre}")


# Функция для быстрого запуска пайплайна
def run_pipeline(
    subset: str = "medium",
    min_samples_per_genre: int = 100,
    force_reload: bool = False,
    check_audio: bool = True
) -> Dict[str, Any]:
    """
    Быстрый запуск пайплайна с настройками по умолчанию

    Args:
        subset: 'small', 'medium', 'large'
        min_samples_per_genre: минимальное количество треков на жанр
        force_reload: принудительно перезапустить
        check_audio: проверять доступность аудио

    Returns:
        Словарь с подготовленными данными
    """
    pipeline = DataPipeline(
        subset=subset,
        min_samples_per_genre=min_samples_per_genre,
        check_audio_availability=check_audio
    )
    return pipeline.run(force_reload=force_reload)


if __name__ == "__main__":
    # Тестовый запуск
    data = run_pipeline(subset="small", min_samples_per_genre=50, force_reload=True)

    print("\n📋 Доступные ключи в данных:")
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"   {key}: {data[key].shape}")
        elif isinstance(data[key], dict):
            print(f"   {key}: dict with {len(data[key])} keys")
        else:
            print(f"   {key}: {type(data[key]).__name__}")