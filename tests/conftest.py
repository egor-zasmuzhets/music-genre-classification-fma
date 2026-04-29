"""
tests/conftest.py
Pytest фикстуры для тестирования на реальных данных FMA Small
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.config import paths
from src.data.loader import FMALoader
from src.data.preprocessor import DataPreprocessor
from src.data.pipeline import DataPipeline


@pytest.fixture(scope="session")
def fma_small_loader():
    """
    Загрузчик для FMA Small (реальные данные)
    """
    loader = FMALoader()
    return loader


@pytest.fixture(scope="session")
def fma_small_tracks(fma_small_loader):
    """
    Треки из FMA Small
    """
    tracks = fma_small_loader.get_tracks_by_subset("small")
    # Оставляем только треки с жанром
    genre_col = ('track', 'genre_top')
    tracks_with_genre = tracks[tracks[genre_col].notna()].copy()
    return tracks_with_genre


@pytest.fixture(scope="session")
def fma_small_features(fma_small_loader):
    """
    Признаки из FMA Small
    """
    return fma_small_loader.features


@pytest.fixture(scope="session")
def fma_small_pipeline():
    """
    DataPipeline для FMA Small с min_samples_per_genre=50
    (меньше, чем 100, чтобы сохранить больше жанров для тестов)
    """
    pipeline = DataPipeline(
        subset="small",
        min_samples_per_genre=50,
        use_features=True
    )
    return pipeline


@pytest.fixture(scope="session")
def fma_small_processed_data(fma_small_pipeline):
    """
    Полностью обработанные данные FMA Small
    (запускает пайплайн один раз для всех тестов)
    """
    data = fma_small_pipeline.run(force_reload=False)
    return data


@pytest.fixture
def sample_track_ids(fma_small_tracks):
    """
    Несколько ID треков для тестов
    """
    return fma_small_tracks.index[:10].tolist()


@pytest.fixture
def genre_col():
    """Колонка с жанрами"""
    return ('track', 'genre_top')