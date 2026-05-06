"""
src/data/config.py
Управление конфигурацией проекта.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# ============================================================================
# 1. ОПРЕДЕЛЕНИЕ КОРНЯ ПРОЕКТА
# ============================================================================

def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Находит корень проекта по наличию configs/paths.yaml"""
    if start_path is None:
        start_path = Path(__file__).parent.parent.parent

    current = Path(start_path).resolve()

    while current != current.parent:
        if (current / "configs" / "paths.yaml").exists():
            return current
        current = current.parent

    return Path(start_path).resolve()


PROJECT_ROOT = find_project_root()


# ============================================================================
# 2. БАЗОВЫЙ КЛАСС ДЛЯ КОНФИГОВ
# ============================================================================

class Config:
    """Загрузка и кэширование YAML конфигов"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            self.config_dir = PROJECT_ROOT / "configs"
        else:
            self.config_dir = PROJECT_ROOT / config_dir

        self._cache: Dict[str, Any] = {}

    def load(self, filename: str) -> Dict[str, Any]:
        """Загружает YAML файл из configs/"""
        if filename not in self._cache:
            filepath = self.config_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Конфиг не найден: {filepath}")

            with open(filepath, 'r', encoding='utf-8') as f:
                self._cache[filename] = yaml.safe_load(f)

        return self._cache[filename]

    def reload(self):
        self._cache.clear()


_config = Config()


# ============================================================================
# 3. ПУТИ (из paths.yaml)
# ============================================================================

class Paths:
    """Управление путями проекта"""

    def __init__(self):
        self._raw = _config.load("paths.yaml")
        self._resolved: Dict[str, Path] = {}

    def _resolve(self, path_str: str) -> Path:
        """Преобразует строку в абсолютный путь"""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _get_dir(self, key: str) -> Path:
        """Возвращает путь из project_dirs по ключу (поддерживает data.raw)"""
        if key in self._resolved:
            return self._resolved[key]

        project_dirs = self._raw.get("project_dirs", {})

        if "." in key:
            parts = key.split(".")
            value = project_dirs
            for part in parts:
                value = value.get(part, {})
            path_str = value if isinstance(value, str) else ""
        else:
            path_str = project_dirs.get(key, "")

        resolved = self._resolve(path_str) if path_str else PROJECT_ROOT / key
        self._resolved[key] = resolved
        return resolved

    # ========================================================================
    # Внешние данные
    # ========================================================================

    @property
    def metadata_dir(self) -> Path:
        return Path(self._raw["external_data"]["metadata_dir"])

    @property
    def active_zip(self) -> Path:
        return Path(self._raw["external_data"]["active_zip"])

    @property
    def active_subset(self) -> str:
        return self._raw["external_data"]["active_subset"]

    def get_fma_zip(self, subset: str) -> Path:
        return Path(self._raw["external_data"][f"fma_{subset}_zip"])

    # ========================================================================
    # Внутренние директории
    # ========================================================================

    @property
    def raw_data_dir(self) -> Path:
        return self._get_dir("data.raw")

    @property
    def processed_data_dir(self) -> Path:
        return self._get_dir("data.processed")

    @property
    def external_data_dir(self) -> Path:
        return self._get_dir("data.external")

    @property
    def checkpoints_dir(self) -> Path:
        return self._get_dir("checkpoints")

    @property
    def results_dir(self) -> Path:
        return self._get_dir("results")

    @property
    def models_dir(self) -> Path:
        return self._get_dir("models")

    @property
    def logs_dir(self) -> Path:
        return self._get_dir("logs")

    @property
    def configs_dir(self) -> Path:
        """Директория с конфигами"""
        return PROJECT_ROOT / "configs"

    # ========================================================================
    # XGBoost Mono Classification (Single-Label)
    # ========================================================================

    @property
    def xgboost_mono_dir(self) -> Path:
        """Директория для результатов XGBoost mono"""
        path = self.results_dir / "xgboost" / "mono"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def xgboost_mono_models_dir(self) -> Path:
        """Директория для моделей XGBoost mono"""
        path = self.xgboost_mono_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def xgboost_mono_plots_dir(self) -> Path:
        """Директория для графиков XGBoost mono"""
        path = self.xgboost_mono_dir / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def xgboost_mono_grid_search_dir(self) -> Path:
        """Директория для результатов Grid Search"""
        path = self.xgboost_mono_dir / "grid_search"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def xgboost_mono_metrics_dir(self) -> Path:
        """Директория для JSON с метриками"""
        path = self.xgboost_mono_dir / "metrics"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ========================================================================
    # XGBoost Multi Classification (Multi-Label) — для будущего
    # ========================================================================

    @property
    def xgboost_multi_dir(self) -> Path:
        """Директория для результатов XGBoost multi"""
        path = self.results_dir / "xgboost" / "multi"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def xgboost_multi_models_dir(self) -> Path:
        """Директория для моделей XGBoost multi"""
        path = self.xgboost_multi_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def xgboost_multi_plots_dir(self) -> Path:
        """Директория для графиков XGBoost multi"""
        path = self.xgboost_multi_dir / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def xgboost_multi_metrics_dir(self) -> Path:
        """Директория для JSON с метриками"""
        path = self.xgboost_multi_dir / "metrics"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ========================================================================
    # Аудио фичи (MFCC, спектрограммы) — кэш для CNN
    # ========================================================================

    @property
    def audio_features_dir(self) -> Path:
        """Директория для кэшированных аудио-фич"""
        path = self.processed_data_dir / "audio_features"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def mfcc_cache_dir(self) -> Path:
        """Директория для кэша MFCC"""
        path = self.audio_features_dir / "mfcc"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def spectrogram_cache_dir(self) -> Path:
        """Директория для кэша спектрограмм"""
        path = self.audio_features_dir / "spectrograms"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def waveform_cache_dir(self) -> Path:
        """Директория для кэша"""
        path = self.audio_features_dir / "waveforms"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def audio_metadata_file(self) -> Path:
        """Файл с метаданными аудио (какие треки закэшированы)"""
        return self.audio_features_dir / "audio_metadata.json"

    # ========================================================================
    # Конкретные файлы
    # ========================================================================

    def get_tracks_csv(self) -> Path:
        return self.metadata_dir / "tracks.csv"

    def get_features_csv(self) -> Path:
        return self.metadata_dir / "features.csv"

    def get_genres_csv(self) -> Path:
        return self.metadata_dir / "genres.csv"

    # ========================================================================
    # Вспомогательные методы
    # ========================================================================

    def ensure_dirs(self):
        """Создаёт все необходимые директории"""
        for d in [self.raw_data_dir, self.processed_data_dir, self.external_data_dir,
                  self.checkpoints_dir, self.results_dir, self.models_dir, self.logs_dir,
                  self.xgboost_mono_dir, self.xgboost_mono_models_dir,
                  self.xgboost_mono_plots_dir, self.xgboost_mono_grid_search_dir,
                  self.xgboost_mono_metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def print_info(self):
        """Выводит информацию о путях"""
        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"\n[Внешние данные]")
        print(f"  metadata_dir: {self.metadata_dir}")
        print(f"  active_zip:   {self.active_zip}")
        print(f"\n[Внутренние директории]")
        print(f"  raw_data:     {self.raw_data_dir}")
        print(f"  processed:    {self.processed_data_dir}")
        print(f"  checkpoints:  {self.checkpoints_dir}")
        print(f"  results:      {self.results_dir}")
        print(f"  models:       {self.models_dir}")
        print(f"\n[XGBoost Mono]")
        print(f"  models_dir:   {self.xgboost_mono_models_dir}")
        print(f"  plots_dir:    {self.xgboost_mono_plots_dir}")
        print(f"  metrics_dir:  {self.xgboost_mono_metrics_dir}")
        print(f"  grid_search:  {self.xgboost_mono_grid_search_dir}")


# ============================================================================
# 4. ПАРАМЕТРЫ АУДИО (из audio.yaml)
# ============================================================================

class AudioParams:
    """Параметры извлечения аудио-признаков"""

    def __init__(self):
        self._raw = _config.load("audio.yaml")

    @property
    def sample_rate(self) -> int:
        return self._raw.get("sample_rate", 22050)

    @property
    def duration(self) -> int:
        return self._raw.get("duration", 30)

    @property
    def n_fft(self) -> int:
        return self._raw.get("n_fft", 2048)

    @property
    def hop_length(self) -> int:
        return self._raw.get("hop_length", 512)

    @property
    def win_length(self) -> int:
        return self._raw.get("win_length", 2048)

    @property
    def n_mels(self) -> int:
        return self._raw.get("n_mels", 128)

    @property
    def fmin(self) -> int:
        return self._raw.get("fmin", 0)

    @property
    def fmax(self) -> int:
        return self._raw.get("fmax", 8000)

    @property
    def n_mfcc(self) -> int:
        return self._raw.get("n_mfcc", 20)

    @property
    def n_chroma(self) -> int:
        return self._raw.get("n_chroma", 12)

    @property
    def n_bands(self) -> int:
        return self._raw.get("n_bands", 7)

    @property
    def augmentation(self) -> dict:
        return self._raw.get("augmentation", {})

    def to_dict(self) -> dict:
        """Возвращает все параметры словарём (удобно для передачи в функции)"""
        return {
            "sr": self.sample_rate,
            "duration": self.duration,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "n_mfcc": self.n_mfcc,
        }

    def print_info(self):
        print("[Аудио параметры]")
        print(f"  sample_rate:  {self.sample_rate}")
        print(f"  duration:     {self.duration} sec")
        print(f"  n_mels:       {self.n_mels}")
        print(f"  n_mfcc:       {self.n_mfcc}")
        print(f"  hop_length:   {self.hop_length}")


# ============================================================================
# 5. ГЛОБАЛЬНЫЕ ЭКЗЕМПЛЯРЫ
# ============================================================================

paths = Paths()
audio_params = AudioParams()


# ============================================================================
# 6. ТЕСТОВЫЙ ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    paths.print_info()
    print()
    audio_params.print_info()

    print("\nСоздание директорий...")
    paths.ensure_dirs()
    print("✅ Готово")