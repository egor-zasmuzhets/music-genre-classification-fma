"""
src/data/audio_loader.py
Загрузка аудио из ZIP-архивов FMA
"""

import zipfile
import io
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
import hashlib

from src.data.config import paths, audio_params


class AudioLoader:
    """
    Загрузчик аудио из ZIP-архивов FMA.

    Особенности:
    - Читает MP3 напрямую из ZIP без распаковки
    - Кэширует загруженные треки (опционально)
    - Поддерживает обрезку до нужной длительности
    - Может сохранять кэш на диск для повторного использования
    - Возвращает статус загрузки
    """

    def __init__(
            self,
            zip_path: Optional[Path] = None,
            cache_size: int = 100,
            use_cache: bool = True,
            use_disk_cache: bool = False,
            disk_cache_dir: Optional[Path] = None
    ):
        """
        Args:
            zip_path: путь к ZIP архиву (если None — из config)
            cache_size: размер кэша в памяти (количество треков)
            use_cache: использовать ли кэш в памяти
            use_disk_cache: сохранять ли аудио на диск
            disk_cache_dir: директория для дискового кэша
        """
        self.zip_path = zip_path or paths.active_zip
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.use_disk_cache = use_disk_cache
        self.disk_cache_dir = disk_cache_dir or paths.waveform_cache_dir

        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP архив не найден: {self.zip_path}")

        # Открываем ZIP архив
        self._zip = None
        self._cached_audio: Dict[Tuple, np.ndarray] = {}

        # Создаём директорию для дискового кэша
        if self.use_disk_cache:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ AudioLoader готов: {self.zip_path}")
        print(f"   Размер архива: {self.zip_path.stat().st_size / 1024 ** 3:.2f} GB")
        if use_disk_cache:
            print(f"   Дисковый кэш: {self.disk_cache_dir}")

    @property
    def zip(self):
        """Ленивое открытие ZIP архива"""
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, 'r')
        return self._zip

    def get_audio_path(self, track_id: int) -> str:
        """
        Возвращает путь к MP3 файлу внутри ZIP

        Формат FMA: fma_medium/001/001.mp3 (треки разбиты по папкам 000-099)
        """
        track_str = str(track_id).zfill(6)
        folder = track_str[:3]
        return f"fma_medium/{folder}/{track_str}.mp3"

    def _get_disk_cache_path(self, track_id: int, sr: int, duration: float) -> Path:
        """Возвращает путь для дискового кэша"""
        key = f"track_{track_id}_sr_{sr}_dur_{duration}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.disk_cache_dir / f"{hash_key}.npy"

    def _load_from_disk_cache(
        self,
        track_id: int,
        sr: int,
        duration: float
    ) -> Optional[np.ndarray]:
        """Загружает аудио из дискового кэша"""
        if not self.use_disk_cache:
            return None

        cache_path = self._get_disk_cache_path(track_id, sr, duration)
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                pass
        return None

    def _save_to_disk_cache(
        self,
        audio: np.ndarray,
        track_id: int,
        sr: int,
        duration: float
    ):
        """Сохраняет аудио в дисковый кэш"""
        if not self.use_disk_cache:
            return

        cache_path = self._get_disk_cache_path(track_id, sr, duration)
        try:
            np.save(cache_path, audio)
        except Exception:
            pass

    def load_audio_with_status(
            self,
            track_id: int,
            sr: Optional[int] = None,
            duration: Optional[float] = None,
            offset: float = 0.0,
            use_cache: bool = True
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Загружает аудио для указанного трека с возвратом статуса

        Returns:
            (audio, status_dict)
            status_dict содержит:
            - success: bool
            - error_type: str (если ошибка)
            - error_message: str (если ошибка)
            - loaded_from_cache: bool
        """
        sr = sr or audio_params.sample_rate
        duration = duration or audio_params.duration

        status = {
            'success': False,
            'error_type': None,
            'error_message': None,
            'loaded_from_cache': False,
            'track_id': track_id,
            'sr': sr,
            'duration': duration
        }

        # Проверяем дисковый кэш
        if self.use_disk_cache:
            audio = self._load_from_disk_cache(track_id, sr, duration)
            if audio is not None:
                status['success'] = True
                status['loaded_from_cache'] = True
                status['cache_type'] = 'disk'
                return audio, status

        # Проверяем кэш в памяти
        cache_key = (track_id, sr, duration, offset)
        if use_cache and self.use_cache and cache_key in self._cached_audio:
            status['success'] = True
            status['loaded_from_cache'] = True
            status['cache_type'] = 'memory'
            return self._cached_audio[cache_key], status

        try:
            audio_path = self.get_audio_path(track_id)

            # Пробуем разные варианты пути если файл не найден
            if audio_path not in self.zip.namelist():
                alt_path = f"fma_medium/{track_id:06d}.mp3"
                if alt_path in self.zip.namelist():
                    audio_path = alt_path
                else:
                    status['error_type'] = 'not_found'
                    status['error_message'] = f"Трек {track_id} не найден в архиве"
                    return None, status

            with self.zip.open(audio_path) as f:
                audio_bytes = f.read()

            audio, _ = librosa.load(
                io.BytesIO(audio_bytes),
                sr=sr,
                duration=duration,
                offset=offset,
                res_type='kaiser_fast'
            )

            expected_length = int(sr * duration)
            if len(audio) < expected_length:
                audio = np.pad(audio, (0, expected_length - len(audio)))
            else:
                audio = audio[:expected_length]

            # Сохраняем в дисковый кэш
            if self.use_disk_cache:
                self._save_to_disk_cache(audio, track_id, sr, duration)

            # Сохраняем в кэш памяти
            if use_cache and self.use_cache and len(self._cached_audio) < self.cache_size:
                self._cached_audio[cache_key] = audio

            status['success'] = True
            return audio, status

        except KeyError as e:
            status['error_type'] = 'not_found'
            status['error_message'] = f"Трек {track_id} не найден в архиве: {e}"
            return None, status
        except Exception as e:
            status['error_type'] = 'processing_error'
            status['error_message'] = str(e)
            return None, status

    def load_audio(
            self,
            track_id: int,
            sr: Optional[int] = None,
            duration: Optional[float] = None,
            offset: float = 0.0,
            use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Загружает аудио (совместимость со старым кодом)

        Returns:
            numpy array с аудиосигналом или None при ошибке
        """
        audio, _ = self.load_audio_with_status(track_id, sr, duration, offset, use_cache)
        return audio

    def load_audio_batch_with_status(
        self,
        track_ids: List[int],
        sr: Optional[int] = None,
        duration: Optional[float] = None,
        offset: float = 0.0,
        verbose: bool = True
    ) -> Tuple[Dict[int, Optional[np.ndarray]], Dict[int, Dict[str, Any]]]:
        """
        Загружает несколько треков пакетно с возвратом статуса

        Returns:
            (audio_dict, status_dict)
        """
        audio_results = {}
        status_results = {}
        total = len(track_ids)
        failed = []

        for i, track_id in enumerate(track_ids):
            if verbose and (i + 1) % 100 == 0:
                print(f"   Загружено {i + 1}/{total} треков (неудачных: {len(failed)})")

            audio, status = self.load_audio_with_status(track_id, sr, duration, offset)
            audio_results[track_id] = audio
            status_results[track_id] = status

            if not status['success']:
                failed.append(track_id)

        if verbose and failed:
            print(f"   ⚠️ Не удалось загрузить {len(failed)} треков: {failed[:10]}{'...' if len(failed) > 10 else ''}")

        return audio_results, status_results

    def get_available_tracks(self) -> List[int]:
        """
        Возвращает список доступных треков в ZIP архиве
        """
        tracks = []
        for name in self.zip.namelist():
            if name.endswith('.mp3') and 'fma_medium' in name:
                parts = name.replace('.mp3', '').split('/')
                if len(parts) >= 3:
                    try:
                        track_id = int(parts[-1])
                        tracks.append(track_id)
                    except ValueError:
                        pass
        return sorted(tracks)

    def get_failed_tracks_report(self, status_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Формирует отчёт о неудачных треках"""
        failed_by_type = {}
        failed_tracks = []

        for track_id, status in status_results.items():
            if not status['success']:
                failed_tracks.append(track_id)
                error_type = status['error_type']
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

    def close(self):
        """Закрывает ZIP архив"""
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def clear_cache(self):
        """Очищает кэш в памяти"""
        self._cached_audio.clear()

    def clear_disk_cache(self):
        """Очищает дисковый кэш"""
        if self.disk_cache_dir.exists():
            import shutil
            shutil.rmtree(self.disk_cache_dir)
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Дисковый кэш очищен: {self.disk_cache_dir}")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    loader = AudioLoader()
    tracks = loader.get_available_tracks()
    print(f"Доступно треков: {len(tracks)}")
    print(f"Примеры: {tracks[:10]}")
    loader.close()