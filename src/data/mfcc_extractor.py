"""
src/data/mfcc_extractor.py
Извлечение MFCC признаков из аудио
"""

import numpy as np
import librosa
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path

from src.data.config import paths, audio_params
from src.data.audio_loader import AudioLoader


@dataclass
class MFCCConfig:
    """
    Конфигурация извлечения MFCC

    Параметры из audio.yaml используются по умолчанию.
    Дополнительные параметры (include_delta, include_delta2) специфичны для MFCC.
    """
    include_delta: bool = True
    include_delta2: bool = True

    # Параметры из audio.yaml (будут загружены автоматически, если не указаны)
    n_mfcc: Optional[int] = None
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    win_length: Optional[int] = None
    sr: Optional[int] = None
    duration: Optional[float] = None

    def __post_init__(self):
        """Загружает параметры из audio_params, если не указаны явно"""
        if self.n_mfcc is None:
            self.n_mfcc = audio_params.n_mfcc
        if self.n_fft is None:
            self.n_fft = audio_params.n_fft
        if self.hop_length is None:
            self.hop_length = audio_params.hop_length
        if self.win_length is None:
            self.win_length = audio_params.win_length
        if self.sr is None:
            self.sr = audio_params.sample_rate
        if self.duration is None:
            self.duration = audio_params.duration

    def get_cache_key(self) -> str:
        """Возвращает уникальный ключ для кэша на основе конфига"""
        data = {
            'n_mfcc': self.n_mfcc,
            'include_delta': self.include_delta,
            'include_delta2': self.include_delta2,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'n_fft': self.n_fft,
        }
        key_str = str(sorted(data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает все параметры словарём (для librosa)"""
        return {
            'n_mfcc': self.n_mfcc,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
        }


class MFCCExtractor:
    """
    Извлекает MFCC признаки из аудио и преобразует в формат для CNN

    Поддерживает кэширование на диск для ускорения повторных загрузок
    """

    def __init__(
            self,
            config: Optional[MFCCConfig] = None,
            audio_loader: Optional[AudioLoader] = None,
            use_disk_cache: bool = True,
            disk_cache_dir: Optional[Path] = None
    ):
        self.config = config or MFCCConfig()
        self.audio_loader = audio_loader or AudioLoader()
        self.use_disk_cache = use_disk_cache
        self.disk_cache_dir = disk_cache_dir or paths.mfcc_cache_dir

        # Создаём поддиректорию для этой конфигурации
        if self.use_disk_cache:
            self.cache_subdir = self.disk_cache_dir / self.config.get_cache_key()
            self.cache_subdir.mkdir(parents=True, exist_ok=True)

    def _get_full_features_cache_path(self, track_id: int) -> Path:
        """Возвращает путь для кэша полных признаков"""
        return self.cache_subdir / f"track_{track_id}_full_features.pkl"

    def _load_full_features_from_cache(
        self,
        track_id: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """Загружает полные признаки из кэша"""
        if not self.use_disk_cache:
            return None

        cache_path = self._get_full_features_cache_path(track_id)
        if cache_path.exists():
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _save_full_features_to_cache(
        self,
        track_id: int,
        features: Dict[str, np.ndarray]
    ):
        """Сохраняет полные признаки в кэш"""
        if not self.use_disk_cache:
            return

        cache_path = self._get_full_features_cache_path(track_id)
        try:
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception:
            pass

    def _get_disk_cache_path(self, track_id: int, with_deltas: bool) -> Path:
        """Возвращает путь для дискового кэша MFCC"""
        suffix = "with_deltas" if with_deltas else "base"
        return self.cache_subdir / f"track_{track_id}_{suffix}.npy"

    def _load_from_disk_cache(
        self,
        track_id: int,
        with_deltas: bool
    ) -> Optional[np.ndarray]:
        """Загружает MFCC из дискового кэша"""
        if not self.use_disk_cache:
            return None

        cache_path = self._get_disk_cache_path(track_id, with_deltas)
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                pass
        return None

    def _save_to_disk_cache(
        self,
        mfcc: np.ndarray,
        track_id: int,
        with_deltas: bool
    ):
        """Сохраняет MFCC в дисковый кэш"""
        if not self.use_disk_cache:
            return

        cache_path = self._get_disk_cache_path(track_id, with_deltas)
        try:
            np.save(cache_path, mfcc)
        except Exception:
            pass

    def extract_from_audio(
            self,
            audio: np.ndarray,
            sr: int
    ) -> Dict[str, np.ndarray]:
        """
        Извлекает MFCC из аудиосигнала

        Returns:
            Словарь с признаками:
            - mfcc: (n_mfcc, n_frames)
            - mfcc_delta: (n_mfcc, n_frames) если include_delta
            - mfcc_delta2: (n_mfcc, n_frames) если include_delta2
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            **self.config.to_dict()
        )

        result = {'mfcc': mfcc}

        if self.config.include_delta:
            result['mfcc_delta'] = librosa.feature.delta(mfcc)

        if self.config.include_delta2:
            result['mfcc_delta2'] = librosa.feature.delta(mfcc, order=2)

        return result

    def extract_from_track_id_with_status(
            self,
            track_id: int,
            use_cache: bool = True
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
        """
        Извлекает MFCC для трека с возвратом статуса

        Returns:
            (features, status_dict)
        """
        status = {
            'success': False,
            'error_type': None,
            'error_message': None,
            'track_id': track_id,
            'loaded_from_cache': False
        }

        # Проверяем кэш (для готового результата)
        if use_cache and self.use_disk_cache:
            cached = self._load_full_features_from_cache(track_id)
            if cached is not None:
                status['success'] = True
                status['loaded_from_cache'] = True
                status['cache_type'] = 'disk'
                return cached, status

        audio, audio_status = self.audio_loader.load_audio_with_status(
            track_id, self.config.sr, self.config.duration
        )

        if not audio_status['success']:
            status['error_type'] = audio_status['error_type']
            status['error_message'] = f"Ошибка загрузки аудио: {audio_status['error_message']}"
            return None, status

        try:
            features = self.extract_from_audio(audio, self.config.sr)

            if use_cache and self.use_disk_cache:
                self._save_full_features_to_cache(track_id, features)

            status['success'] = True
            return features, status

        except Exception as e:
            status['error_type'] = 'mfcc_extraction_error'
            status['error_message'] = str(e)
            return None, status

    def extract_from_track_id(
            self,
            track_id: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """Извлекает MFCC для трека по ID (совместимость)"""
        features, _ = self.extract_from_track_id_with_status(track_id)
        return features

    def get_mfcc_matrix_with_status(
            self,
            track_id: int,
            flatten: bool = False,
            use_cache: bool = True
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Возвращает матрицу MFCC с возвратом статуса
        """
        features, status = self.extract_from_track_id_with_status(track_id, use_cache)

        if not status['success']:
            return None, status

        mfcc = features['mfcc']

        if flatten:
            return mfcc.flatten(), status
        return mfcc, status

    def get_mfcc_matrix(
            self,
            track_id: int,
            flatten: bool = False,
            use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """Возвращает матрицу MFCC (совместимость)"""
        mfcc, _ = self.get_mfcc_matrix_with_status(track_id, flatten, use_cache)
        return mfcc

    def get_mfcc_with_deltas_with_status(
            self,
            track_id: int,
            flatten: bool = False,
            use_cache: bool = True
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Возвращает объединённую матрицу MFCC + дельты с возвратом статуса
        """
        if use_cache:
            cached = self._load_from_disk_cache(track_id, with_deltas=True)
            if cached is not None:
                status = {
                    'success': True,
                    'loaded_from_cache': True,
                    'cache_type': 'disk',
                    'track_id': track_id
                }
                combined = cached
                if flatten:
                    return combined.flatten(), status
                return combined, status

        features, status = self.extract_from_track_id_with_status(track_id, use_cache)

        if not status['success']:
            return None, status

        matrices = [features['mfcc']]
        if self.config.include_delta and 'mfcc_delta' in features:
            matrices.append(features['mfcc_delta'])
        if self.config.include_delta2 and 'mfcc_delta2' in features:
            matrices.append(features['mfcc_delta2'])

        combined = np.vstack(matrices)

        # Кэшируем
        if self.use_disk_cache:
            self._save_to_disk_cache(combined, track_id, with_deltas=True)

        if flatten:
            return combined.flatten(), status
        return combined, status

    def get_mfcc_with_deltas(
            self,
            track_id: int,
            flatten: bool = False,
            use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """Возвращает объединённую матрицу MFCC + дельты (совместимость)"""
        mfcc, _ = self.get_mfcc_with_deltas_with_status(track_id, flatten, use_cache)
        return mfcc

    def prepare_for_cnn_with_status(
            self,
            track_id: int,
            target_frames: int = 128,
            use_cache: bool = True
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Подготавливает MFCC для подачи в CNN с возвратом статуса
        """
        combined, status = self.get_mfcc_with_deltas_with_status(track_id, flatten=False, use_cache=use_cache)

        if not status['success']:
            return None, status

        if combined is None:
            status['success'] = False
            status['error_type'] = 'no_data'
            status['error_message'] = "Нет данных MFCC"
            return None, status

        combined = combined.astype(np.float32)

        n_channels, n_frames = combined.shape

        if n_frames < target_frames:
            pad_width = ((0, 0), (0, target_frames - n_frames))
            combined = np.pad(combined, pad_width, mode='constant')
        else:
            start = (n_frames - target_frames) // 2
            combined = combined[:, start:start + target_frames]

        return combined, status

    def prepare_for_cnn(
            self,
            track_id: int,
            target_frames: int = 128
    ) -> Optional[np.ndarray]:
        """Подготавливает MFCC для подачи в CNN (совместимость)"""
        mfcc, _ = self.prepare_for_cnn_with_status(track_id, target_frames)
        return mfcc

    def extract_batch_with_status(
        self,
        track_ids: List[int],
        target_frames: int = 128,
        verbose: bool = True
    ) -> Tuple[Dict[int, Optional[np.ndarray]], Dict[int, Dict[str, Any]]]:
        """
        Извлекает MFCC для списка треков пакетно с возвратом статуса

        Returns:
            (results_dict, status_dict)
        """
        results = {}
        status_results = {}
        total = len(track_ids)
        failed = []

        for i, track_id in enumerate(track_ids):
            if verbose and (i + 1) % 50 == 0:
                print(f"   Обработано {i + 1}/{total} треков (неудачных: {len(failed)})")

            mfcc, status = self.prepare_for_cnn_with_status(track_id, target_frames)
            results[track_id] = mfcc
            status_results[track_id] = status

            if not status['success']:
                failed.append(track_id)

        if verbose and failed:
            print(f"   ⚠️ Не удалось обработать {len(failed)} треков: {failed[:10]}{'...' if len(failed) > 10 else ''}")

        return results, status_results

    def get_failed_tracks_report(self, status_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Формирует отчёт о неудачных треках"""
        failed_by_type = {}
        failed_tracks = []

        for track_id, status in status_results.items():
            if not status['success']:
                failed_tracks.append(track_id)
                error_type = status.get('error_type', 'unknown')
                if error_type not in failed_by_type:
                    failed_by_type[error_type] = []
                failed_by_type[error_type].append(track_id)

        return {
            'total_processed': len(status_results),
            'failed_count': len(failed_tracks),
            'failed_tracks': failed_tracks,
            'failed_by_type': failed_by_type,
            'success_rate': (len(status_results) - len(failed_tracks)) / len(status_results) if status_results else 0
        }

    def clear_cache(self):
        """Очищает дисковый кэш для текущей конфигурации"""
        if self.cache_subdir.exists():
            import shutil
            shutil.rmtree(self.cache_subdir)
            self.cache_subdir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Кэш MFCC очищен: {self.cache_subdir}")


if __name__ == "__main__":
    extractor = MFCCExtractor()
    test_tracks = [2, 3, 5, 10, 20]
    for track_id in test_tracks:
        mfcc = extractor.prepare_for_cnn(track_id, target_frames=128)
        if mfcc is not None:
            print(f"Трек {track_id}: MFCC shape = {mfcc.shape}")
        else:
            print(f"Трек {track_id}: не найден")