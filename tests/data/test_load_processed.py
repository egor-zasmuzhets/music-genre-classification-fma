"""
tests/data/test_load_processed.py
Тесты для LoadProcessedData на реальных данных FMA Small
"""

import pytest
import numpy as np
from src.data.load_processed import LoadProcessedData, load_data


class TestLoadProcessedData:
    """Тесты загрузчика подготовленных данных"""

    def test_load_processed_exists(self, fma_small_processed_data, fma_small_pipeline):
        """Проверка, что данные существуют"""
        loader = LoadProcessedData(
            subset="small",
            min_samples_per_genre=50,
            cache_dir=fma_small_pipeline.cache_dir
        )

        assert loader.exists() is True

    def test_load_processed_data(self, fma_small_processed_data, fma_small_pipeline):
        """Проверка загрузки данных"""
        loader = LoadProcessedData(
            subset="small",
            min_samples_per_genre=50,
            cache_dir=fma_small_pipeline.cache_dir
        )

        data = loader.load()

        # Проверяем структуру
        expected_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
                         'label_encoder', 'scaler', 'genre_names', 'metadata']
        for key in expected_keys:
            assert key in data

        # Проверяем, что данные соответствуют оригинальным
        np.testing.assert_array_equal(data['X_train'], fma_small_processed_data['X_train'])
        np.testing.assert_array_equal(data['y_train'], fma_small_processed_data['y_train'])

        # Проверяем genre_names
        assert data['genre_names'] == fma_small_processed_data['genre_names']

    def test_load_data_function(self, fma_small_processed_data, monkeypatch):
        """Проверка функции-хелпера load_data"""
        # Функция load_data полагается на paths.active_subset
        # Мокаем, чтобы использовать small
        data = load_data(subset="small", min_samples_per_genre=50)

        assert data is not None
        assert 'X_train' in data
        assert len(data['X_train']) > 0

    def test_load_data_as_dataframe(self, fma_small_processed_data, fma_small_pipeline):
        """Проверка загрузки в формате DataFrame"""
        loader = LoadProcessedData(
            subset="small",
            min_samples_per_genre=50,
            cache_dir=fma_small_pipeline.cache_dir
        )

        data = loader.load_to_dataframe

        # Проверяем типы
        import pandas as pd
        assert isinstance(data['X_train'], (pd.DataFrame, np.ndarray))
        assert isinstance(data['y_train'], np.ndarray)

    def test_load_processed_print_info(self, fma_small_processed_data, fma_small_pipeline, capsys):
        """Проверка вывода информации (без ошибок)"""
        loader = LoadProcessedData(
            subset="small",
            min_samples_per_genre=50,
            cache_dir=fma_small_pipeline.cache_dir
        )

        loader.print_info()

        captured = capsys.readouterr()
        assert "ЗАГРУЖЕННЫЕ ДАННЫЕ" in captured.out

    def test_load_processed_file_not_found(self, tmp_path):
        """Проверка поведения при отсутствии данных"""
        loader = LoadProcessedData(
            subset="small",
            min_samples_per_genre=50,
            cache_dir=tmp_path
        )

        assert loader.exists() is False

        with pytest.raises(FileNotFoundError):
            loader.load()