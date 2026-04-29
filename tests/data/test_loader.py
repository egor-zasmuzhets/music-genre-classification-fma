"""
tests/data/test_loader.py
Тесты для FMALoader на реальных данных FMA Small
"""

import pytest
import pandas as pd
from src.data.loader import FMALoader


class TestFMALoader:
    """Тесты загрузчика метаданных FMA"""

    def test_loader_initialization(self, fma_small_loader):
        """Проверка инициализации загрузчика"""
        assert fma_small_loader is not None
        assert fma_small_loader.metadata_dir.exists()

    def test_tracks_loading(self, fma_small_loader):
        """Проверка загрузки tracks.csv"""
        tracks = fma_small_loader.tracks

        # Проверяем, что данные загрузились
        assert isinstance(tracks, pd.DataFrame)
        assert len(tracks) > 0

        # Проверяем наличие необходимых колонок
        expected_columns = [('track', 'genre_top'), ('set', 'subset'), ('set', 'split')]
        for col in expected_columns:
            assert col in tracks.columns, f"Колонка {col} не найдена"

    def test_features_loading(self, fma_small_loader):
        """Проверка загрузки features.csv"""
        features = fma_small_loader.features

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

        # В features.csv должно быть много признаков (обычно 518)
        # Проверяем, что есть хотя бы несколько MFCC колонок
        assert features.shape[1] > 100, f"Слишком мало признаков: {features.shape[1]}"

    def test_genres_loading(self, fma_small_loader):
        """Проверка загрузки genres.csv"""
        genres = fma_small_loader.genres

        assert isinstance(genres, pd.DataFrame)
        assert 'title' in genres.columns
        assert len(genres) > 0

    def test_get_tracks_by_subset_small(self, fma_small_loader):
        """Проверка фильтрации по подмножеству 'small'"""
        small_tracks = fma_small_loader.get_tracks_by_subset("small")

        # Проверяем, что все треки из подмножества small
        subset_col = ('set', 'subset')
        if subset_col in small_tracks.columns:
            unique_subsets = small_tracks[subset_col].unique()
            assert all(s == 'small' for s in unique_subsets)

    def test_get_tracks_by_subset_medium(self, fma_small_loader):
        """Проверка фильтрации по подмножеству 'medium'"""
        medium_tracks = fma_small_loader.get_tracks_by_subset("medium")

        # FMA Small не должен содержать medium треков, но метод должен работать
        assert isinstance(medium_tracks, pd.DataFrame)

    def test_get_available_splits(self, fma_small_tracks, fma_small_loader):
        """Проверка получения разбиения train/val/test"""
        splits = fma_small_loader.get_available_splits(fma_small_tracks)

        expected_splits = ['training', 'validation', 'test']
        for split in expected_splits:
            assert split in splits, f"Сплит {split} не найден"

        # Проверяем, что индексы не пустые
        for split in expected_splits:
            assert len(splits[split]) > 0, f"Сплит {split} пуст"

        # Проверяем, что все треки распределены
        total = sum(len(splits[s]) for s in expected_splits)
        # Допускаем, что некоторые треки могут быть без разметки
        assert total <= len(fma_small_tracks)

    def test_get_genre_mapping(self, fma_small_loader):
        """Проверка маппинга ID жанров в названия"""
        genre_mapping = fma_small_loader.get_genre_mapping()

        assert isinstance(genre_mapping, dict)
        assert len(genre_mapping) > 0
        # Проверяем, что значения — строки (названия жанров)
        assert all(isinstance(v, str) for v in genre_mapping.values())

    def test_tracks_have_genres(self, fma_small_tracks):
        """Проверка, что у треков есть жанры"""
        genre_col = ('track', 'genre_top')

        # Проверяем, что жанры не пустые
        tracks_with_genre = fma_small_tracks[fma_small_tracks[genre_col].notna()]
        assert len(tracks_with_genre) > 0, "Нет треков с заполненными жанрами"

        # Проверяем, что жанры — это строки
        genres = tracks_with_genre[genre_col].dropna()
        assert all(isinstance(g, str) for g in genres)

    def test_tracks_have_splits(self, fma_small_tracks):
        """Проверка, что у треков есть разбиение"""
        split_col = ('set', 'split')

        tracks_with_split = fma_small_tracks[fma_small_tracks[split_col].notna()]
        assert len(tracks_with_split) > 0, "Нет треков с разбиением"

        # Проверяем допустимые значения split
        valid_splits = {'training', 'validation', 'test'}
        splits = set(tracks_with_split[split_col].unique())
        assert splits.issubset(valid_splits), f"Недопустимые значения split: {splits - valid_splits}"