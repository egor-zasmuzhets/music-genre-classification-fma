"""
src/data/torch_dataset.py
PyTorch Dataset для загрузки аудио и MFCC с использованием индексов из pipeline
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable
import random

from src.data.load_processed import load_data, load_track_indices
from src.data.mfcc_extractor import MFCCExtractor, MFCCConfig
from src.data.audio_loader import AudioLoader


class MFCCAugmentation:
    """
    Аугментация данных для MFCC

    Поддерживает:
    - Time masking (вырезание временных отрезков)
    - Frequency masking (вырезание частотных полос)
    - Добавление шума
    """

    @staticmethod
    def time_masking(
        mfcc: np.ndarray,
        max_mask_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Маскирует случайный временной отрезок

        Args:
            mfcc: (channels, height, width)
            max_mask_ratio: максимальная доля маскируемой ширины
        """
        _, _, width = mfcc.shape
        mask_len = int(width * random.uniform(0.05, max_mask_ratio))
        mask_len = max(1, min(mask_len, width - 2))
        mask_start = random.randint(0, width - mask_len - 1) if width > mask_len + 1 else 0

        mfcc_aug = mfcc.copy()
        mfcc_aug[:, :, mask_start:mask_start + mask_len] = 0
        return mfcc_aug

    @staticmethod
    def frequency_masking(
        mfcc: np.ndarray,
        max_mask_ratio: float = 0.15
    ) -> np.ndarray:
        """
        Маскирует случайную частотную полосу

        Args:
            mfcc: (channels, height, width)
            max_mask_ratio: максимальная доля маскируемой высоты
        """
        _, height, _ = mfcc.shape
        mask_len = int(height * random.uniform(0.05, max_mask_ratio))
        mask_len = max(1, min(mask_len, height - 2))
        mask_start = random.randint(0, height - mask_len - 1) if height > mask_len + 1 else 0

        mfcc_aug = mfcc.copy()
        mfcc_aug[:, mask_start:mask_start + mask_len, :] = 0
        return mfcc_aug

    @staticmethod
    def add_noise(
        mfcc: np.ndarray,
        noise_std: float = 0.005
    ) -> np.ndarray:
        """
        Добавляет гауссовский шум

        Args:
            mfcc: (channels, height, width)
            noise_std: стандартное отклонение шума
        """
        noise = np.random.normal(0, noise_std, mfcc.shape)
        return mfcc + noise

    @classmethod
    def apply_spec_augment(
        cls,
        mfcc: np.ndarray,
        time_mask_ratio: float = 0.1,
        freq_mask_ratio: float = 0.15,
        p: float = 0.5
    ) -> np.ndarray:
        """
        Применяет SpecAugment (time + frequency masking)
        """
        if random.random() > p:
            return mfcc

        mfcc_aug = mfcc.copy()

        if random.random() > 0.5:
            mfcc_aug = cls.time_masking(mfcc_aug, time_mask_ratio)

        if random.random() > 0.5:
            mfcc_aug = cls.frequency_masking(mfcc_aug, freq_mask_ratio)

        return mfcc_aug


class MFCCDataset(Dataset):
    """
    PyTorch Dataset для обучения CNN на MFCC

    Использует индексы треков из pipeline (train_indices, val_indices, test_indices)
    для загрузки аудио и извлечения MFCC.
    """

    def __init__(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        audio_loader: Optional[AudioLoader] = None,
        mfcc_config: Optional[MFCCConfig] = None,
        target_frames: int = 128,
        use_deltas: bool = True,
        cache_mfcc: bool = True,
        use_disk_cache: bool = True,
        augment: bool = False,
        augmentation_p: float = 0.5,
        track_failure_handler: Optional[Callable] = None
    ):
        """
        Args:
            indices: массив индексов треков
            labels: массив меток (соответствует indices)
            audio_loader: загрузчик аудио
            mfcc_config: конфигурация MFCC
            target_frames: целевое количество фреймов по времени
            use_deltas: использовать дельты как отдельные каналы
            cache_mfcc: кэшировать MFCC в памяти
            use_disk_cache: использовать дисковый кэш для MFCC
            augment: применять ли аугментацию (обычно только для train)
            augmentation_p: вероятность применения аугментации
            track_failure_handler: функция для обработки неудачных треков
        """
        # Конвертируем в список для безопасной работы
        self.indices = indices.tolist() if hasattr(indices, 'tolist') else list(indices)
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        self.target_frames = target_frames
        self.use_deltas = use_deltas
        self.cache_mfcc = cache_mfcc
        self.augment = augment
        self.augmentation_p = augmentation_p
        self.track_failure_handler = track_failure_handler

        # Списки для отслеживания проблемных треков
        self.failed_tracks = []
        self.failed_tracks_details = {}

        # Настройка MFCC экстрактора
        if mfcc_config is None:
            mfcc_config = MFCCConfig(
                include_delta=use_deltas,
                include_delta2=use_deltas
            )

        self.extractor = MFCCExtractor(
            config=mfcc_config,
            audio_loader=audio_loader or AudioLoader(),
            use_disk_cache=use_disk_cache
        )

        # Кэш в памяти для MFCC
        self._cache = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mfcc_tensor: (channels, height, width)
            label_tensor: скаляр
        """
        track_id = self.indices[idx]
        label = self.labels[idx]

        # Проверяем кэш
        if self.cache_mfcc and track_id in self._cache:
            mfcc = self._cache[track_id]
        else:
            mfcc, status = self.extractor.prepare_for_cnn_with_status(
                track_id, self.target_frames
            )

            if mfcc is None:
                # Запоминаем неудачный трек
                if track_id not in self.failed_tracks:
                    self.failed_tracks.append(track_id)
                    self.failed_tracks_details[track_id] = {
                        'error_type': status.get('error_type', 'unknown'),
                        'error_message': status.get('error_message', ''),
                        'dataset_split': 'train' if self.augment else 'val/test'
                    }

                    if self.track_failure_handler:
                        self.track_failure_handler(track_id, status)

                # Возвращаем нули вместо ошибки
                n_channels = 1 + (2 if self.use_deltas else 0)
                mfcc = np.zeros((n_channels, 20, self.target_frames), dtype=np.float32)
            else:
                if self.cache_mfcc:
                    self._cache[track_id] = mfcc

        # Аугментация
        if self.augment:
            mfcc = MFCCAugmentation.apply_spec_augment(
                mfcc,
                p=self.augmentation_p
            )
            if random.random() < 0.3:
                mfcc = MFCCAugmentation.add_noise(mfcc, noise_std=0.003)

        # Преобразуем в тензор PyTorch
        mfcc_tensor = torch.from_numpy(mfcc).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mfcc_tensor, label_tensor

    def get_failed_tracks_report(self) -> Dict[str, Any]:
        """Возвращает отчёт о неудачных треках"""
        return {
            'failed_count': len(self.failed_tracks),
            'failed_tracks': self.failed_tracks,
            'failed_details': self.failed_tracks_details,
            'total_tracks': len(self.indices),
            'success_rate': (len(self.indices) - len(self.failed_tracks)) / len(self.indices) if self.indices else 0
        }

    def get_successful_indices(self) -> List[int]:
        """Возвращает список успешно загруженных индексов"""
        return [idx for idx in self.indices if idx not in self.failed_tracks]

    def get_failed_indices(self) -> List[int]:
        """Возвращает список неудачных индексов"""
        return self.failed_tracks.copy()

    def clear_failed_tracks(self):
        """Очищает список неудачных треков"""
        self.failed_tracks = []
        self.failed_tracks_details = {}

    def get_cached_count(self) -> int:
        return len(self._cache)

    def clear_cache(self):
        self._cache.clear()


def create_mfcc_dataloaders(
    subset: str = "medium",
    min_samples_per_genre: int = 100,
    batch_size: int = 32,
    target_frames: int = 128,
    use_deltas: bool = True,
    num_workers: int = 0,
    cache_mfcc: bool = True,
    use_disk_cache: bool = True,
    augment_train: bool = True,
    filter_failed_tracks: bool = True,
    failure_handler: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, Any]]:
    """
    Создаёт DataLoader для train/val/test на основе MFCC

    Returns:
        (train_loader, val_loader, test_loader, genre_names, failure_report)
    """
    # Загружаем метаданные и индексы
    data = load_data(subset=subset, min_samples_per_genre=min_samples_per_genre)
    train_indices, val_indices, test_indices = load_track_indices(
        subset=subset, min_samples_per_genre=min_samples_per_genre
    )

    genre_names = data['genre_names']

    # Конвертируем индексы в списки для безопасной работы
    train_indices_list = train_indices.tolist() if hasattr(train_indices, 'tolist') else list(train_indices)
    val_indices_list = val_indices.tolist() if hasattr(val_indices, 'tolist') else list(val_indices)
    test_indices_list = test_indices.tolist() if hasattr(test_indices, 'tolist') else list(test_indices)

    # Конвертируем метки в списки
    y_train_list = data['y_train'].tolist() if hasattr(data['y_train'], 'tolist') else list(data['y_train'])
    y_val_list = data['y_val'].tolist() if hasattr(data['y_val'], 'tolist') else list(data['y_val'])
    y_test_list = data['y_test'].tolist() if hasattr(data['y_test'], 'tolist') else list(data['y_test'])

    # Проверяем соответствие размеров
    assert len(train_indices_list) == len(y_train_list)
    assert len(val_indices_list) == len(y_val_list)
    assert len(test_indices_list) == len(y_test_list)

    print("=" * 60)
    print("СОЗДАНИЕ MFCC DATALOADERS")
    print("=" * 60)
    print(f"Датасет: FMA {subset.upper()}")
    print(f"Train треков: {len(train_indices_list)}")
    print(f"Val треков:   {len(val_indices_list)}")
    print(f"Test треков:  {len(test_indices_list)}")
    print(f"Жанров: {len(genre_names)}")
    print(f"Target frames: {target_frames}")
    print(f"Use deltas: {use_deltas}")
    print(f"Аугментация train: {augment_train}")
    print("-" * 60)

    # Создаём датасеты
    train_dataset = MFCCDataset(
        indices=train_indices_list,
        labels=y_train_list,
        target_frames=target_frames,
        use_deltas=use_deltas,
        cache_mfcc=cache_mfcc,
        use_disk_cache=use_disk_cache,
        augment=augment_train,
        augmentation_p=0.5,
        track_failure_handler=failure_handler
    )

    val_dataset = MFCCDataset(
        indices=val_indices_list,
        labels=y_val_list,
        target_frames=target_frames,
        use_deltas=use_deltas,
        cache_mfcc=cache_mfcc,
        use_disk_cache=use_disk_cache,
        augment=False,
        augmentation_p=0.0,
        track_failure_handler=failure_handler
    )

    test_dataset = MFCCDataset(
        indices=test_indices_list,
        labels=y_test_list,
        target_frames=target_frames,
        use_deltas=use_deltas,
        cache_mfcc=cache_mfcc,
        use_disk_cache=use_disk_cache,
        augment=False,
        augmentation_p=0.0,
        track_failure_handler=failure_handler
    )

    # Формируем отчёт о неудачных треках
    failure_report = {
        'train': train_dataset.get_failed_tracks_report(),
        'val': val_dataset.get_failed_tracks_report(),
        'test': test_dataset.get_failed_tracks_report()
    }

    if filter_failed_tracks:
        total_failed = (len(train_dataset.failed_tracks) +
                       len(val_dataset.failed_tracks) +
                       len(test_dataset.failed_tracks))
        if total_failed > 0:
            print(f"\n⚠️ Обнаружено {total_failed} проблемных треков:")
            print(f"   Train: {len(train_dataset.failed_tracks)}")
            print(f"   Val:   {len(val_dataset.failed_tracks)}")
            print(f"   Test:  {len(test_dataset.failed_tracks)}")

    # Создаём даталоадеры
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"\n✅ DataLoader созданы:")
    print(f"   Train: {len(train_loader)} батчей по {batch_size}")
    print(f"   Val:   {len(val_loader)} батчей")
    print(f"   Test:  {len(test_loader)} батчей")
    print("=" * 60)

    return train_loader, val_loader, test_loader, genre_names, failure_report


# Тестовая функция
if __name__ == "__main__":
    train_loader, val_loader, test_loader, genres, report = create_mfcc_dataloaders(
        subset="small",
        min_samples_per_genre=50,
        batch_size=4,
        target_frames=64,
        use_deltas=False,
        augment_train=False
    )

    print(f"\n📊 Отчёт о неудачных треках:")
    for split, r in report.items():
        print(f"   {split}: {r['failed_count']} / {r['total_tracks']} ({r['success_rate']:.2%})")

    for mfcc, labels in train_loader:
        print(f"MFCC shape: {mfcc.shape}")
        print(f"Labels: {labels}")
        break