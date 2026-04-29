"""
tests/data/test_pipeline.py
Тесты для DataPipeline на реальных данных FMA Small
"""

import pytest
import numpy as np
from pathlib import Path
from src.data.pipeline import DataPipeline


class TestDataPipeline:
    """Тесты полного пайплайна"""

    def test_pipeline_initialization(self):
        """Проверка инициализации пайплайна"""
        pipeline = DataPipeline(
            subset="small",
            min_samples_per_genre=50,
            use_features=True
        )

        assert pipeline.subset == "small"
        assert pipeline.min_samples_per_genre == 50
        assert pipeline.use_features is True
        assert pipeline.loader is not None
        assert pipeline.preprocessor is not None

    def test_pipeline_run_returns_data(self, fma_small_processed_data):
        """Проверка, что пайплайн возвращает все необходимые данные"""
        data = fma_small_processed_data

        # Проверяем наличие всех ключей
        expected_keys = [
            'X_train', 'X_val', 'X_test',
            'y_train', 'y_val', 'y_test',
            'label_encoder', 'scaler',
            'genre_names', 'metadata'
        ]

        for key in expected_keys:
            assert key in data, f"Ключ {key} отсутствует в данных"

        # Проверяем, что данные не пустые
        assert len(data['X_train']) > 0
        assert len(data['X_val']) > 0
        assert len(data['X_test']) > 0

        # Проверяем соответствие размеров
        assert len(data['X_train']) == len(data['y_train'])
        assert len(data['X_val']) == len(data['y_val'])
        assert len(data['X_test']) == len(data['y_test'])

    def test_pipeline_metadata(self, fma_small_processed_data):
        """Проверка метаданных пайплайна"""
        metadata = fma_small_processed_data['metadata']

        # Проверяем обязательные поля
        assert 'subset' in metadata
        assert 'num_samples' in metadata
        assert 'num_features' in metadata
        assert 'num_classes' in metadata
        assert 'class_names' in metadata
        assert 'train_size' in metadata
        assert 'val_size' in metadata
        assert 'test_size' in metadata

        # Проверяем соответствие
        assert metadata['subset'] == "small"
        assert metadata['num_classes'] == len(metadata['class_names'])
        assert metadata['train_size'] + metadata['val_size'] + metadata['test_size'] == metadata['num_samples']

    def test_pipeline_caching(self, fma_small_pipeline, tmp_path):
        """Проверка кэширования данных"""
        # Меняем директорию кэша для теста
        fma_small_pipeline.cache_dir = tmp_path

        # Первый запуск — создаёт кэш
        data1 = fma_small_pipeline.run(force_reload=True)

        # Проверяем, что кэш создался
        cache_file = tmp_path / f"pipeline_small_min50.pkl"
        assert cache_file.exists()

        # Второй запуск — должен загрузить из кэша
        data2 = fma_small_pipeline.run(force_reload=False)

        # Проверяем, что данные одинаковые
        np.testing.assert_array_equal(data1['X_train'], data2['X_train'])
        np.testing.assert_array_equal(data1['y_train'], data2['y_train'])

    def test_pipeline_class_weights(self, fma_small_processed_data, fma_small_pipeline):
        """Проверка вычисления весов классов"""
        # Подменяем данные в пайплайне
        fma_small_pipeline._data = fma_small_processed_data

        class_weights = fma_small_pipeline.get_class_weights()

        assert isinstance(class_weights, dict)
        assert len(class_weights) == len(fma_small_processed_data['genre_names'])
        assert all(isinstance(k, (int, np.integer)) for k in class_weights.keys())
        assert all(isinstance(v, float) for v in class_weights.values())
        assert all(isinstance(v, float) for v in class_weights.values())

    def test_pipeline_print_summary(self, fma_small_processed_data, fma_small_pipeline, capsys):
        """Проверка вывода сводки (без ошибок)"""
        fma_small_pipeline._data = fma_small_processed_data

        # Должно работать без ошибок
        fma_small_pipeline.print_summary()

        captured = capsys.readouterr()
        assert "СВОДКА ПОДГОТОВЛЕННЫХ ДАННЫХ" in captured.out

    def test_pipeline_feature_shapes(self, fma_small_processed_data):
        """Проверка размерностей признаков"""
        X_train = fma_small_processed_data['X_train']
        X_val = fma_small_processed_data['X_val']
        X_test = fma_small_processed_data['X_test']

        # Все выборки должны иметь одинаковое количество признаков
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

        # Признаки должны быть числами
        assert np.issubdtype(X_train.dtype, np.number)

    def test_pipeline_label_encoder_consistency(self, fma_small_processed_data):
        """Проверка, что label_encoder одинаково кодирует все выборки"""
        label_encoder = fma_small_processed_data['label_encoder']
        genre_names = fma_small_processed_data['genre_names']

        # Декодируем обратно и проверяем соответствие
        decoded_labels = label_encoder.inverse_transform(fma_small_processed_data['y_train'])

        # Все декодированные метки должны быть в списке жанров
        assert all(label in genre_names for label in decoded_labels)