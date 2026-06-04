"""
src.utils.config.py
Project configuration management.

Loads YAML configuration files and provides structured access to:
- Project paths (directories, external data, models)
- Audio processing parameters
- Model-specific directory management

Usage:
    from src.utils.config import paths, audio_params

    # Access project directories
    print(paths.processed_data_dir)
    print(paths.metadata_dir)

    # Access model-specific paths
    model_paths = paths.get_model("xgboost_mono")
    print(model_paths.checkpoints_dir)

    # Access audio parameters
    print(audio_params.sample_rate)
    print(audio_params.n_mels)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import yaml

from src.utils.logging_utils import LoggingMixin, LoggingConfig


logger = logging.getLogger(__name__)


# ============================================================================
# 1. PROJECT ROOT DETECTION
# ============================================================================

def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Locate the project root directory by searching for 'configs/paths.yaml'.

    Traverses upward from the given start path until a directory containing
    the marker file is found. If not found, falls back to the resolved start path.

    Args:
        start_path: Directory to start the search from.
                    Defaults to the grandparent of this file's directory.

    Returns:
        Absolute path to the project root directory.
    """
    if start_path is None:
        start_path = Path(__file__).parent.parent.parent

    current = Path(start_path).resolve()

    while current != current.parent:
        if (current / "configs" / "paths.yaml").exists():
            return current
        current = current.parent

    logger.warning(
        "Project root marker (configs/paths.yaml) not found. "
        f"Falling back to: {start_path}"
    )
    return Path(start_path).resolve()


PROJECT_ROOT = find_project_root()


# ============================================================================
# 2. YAML CONFIGURATION LOADER
# ============================================================================

class Config:
    """
    YAML configuration file loader with caching.

    Loads configuration files from the project's configs directory
    and caches them for subsequent access. Supports cache clearing
    via the reload method.

    Attributes:
        config_dir: Path to the directory containing YAML config files.
        _cache: Internal cache mapping filenames to parsed configurations.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_dir: Relative path from project root to the config directory.
                        Defaults to 'configs'.
        """
        if config_dir is None:
            self.config_dir = PROJECT_ROOT / "configs"
        else:
            self.config_dir = PROJECT_ROOT / config_dir

        self._cache: Dict[str, Any] = {}

    def load(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Reads the specified YAML file from the config directory,
        caches the result, and returns it. Subsequent calls with the
        same filename return the cached configuration.

        Args:
            filename: Name of the YAML file (e.g., 'paths.yaml').

        Returns:
            Parsed YAML configuration as a dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if filename not in self._cache:
            filepath = self.config_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Configuration file not found: {filepath}")

            logger.debug(f"Loading configuration: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                self._cache[filename] = yaml.safe_load(f)

        return self._cache[filename]

    def reload(self) -> None:
        """
        Clear the configuration cache.

        Forces all subsequent load calls to re-read configuration files
        from disk rather than returning cached versions.
        """
        self._cache.clear()
        logger.debug("Configuration cache cleared")


_config = Config()


# ============================================================================
# 3. MODEL-SPECIFIC PATH MANAGEMENT
# ============================================================================

class ModelPaths:
    """
    Manages directory paths for a specific model.

    Provides structured access to model-related directories
    (results, checkpoints, logs, plots, metrics, etc.) and
    creates them automatically if they don't exist.

    Usage:
        paths.get_model("xgboost_mono").results_dir
        paths.get_model("cnn_mfcc_mono").checkpoints_dir
    """

    def __init__(self, model_config: Dict[str, Any], model_name: str):
        """
        Initialize model paths from configuration.

        Args:
            model_config: Model configuration dictionary from models.yaml.
            model_name: Name of the model (key in models.yaml).
        """
        self.model_name = model_name
        self.config = model_config

        path_config = model_config.get("paths", {})

        self.results_dir = self._resolve_path(
            path_config.get("results_dir", f"results/{model_name}")
        )
        self.models_dir = self._resolve_path(
            path_config.get("models_dir", f"models/{model_name}")
        )
        self.logs_dir = self._resolve_path(
            path_config.get("logs_dir", f"logs/{model_name}")
        )

        self._create_dirs()
        logger.debug(f"ModelPaths initialized for '{model_name}': {self.results_dir}")

    def _resolve_path(self, path: Any) -> Path:
        """
        Convert a path string or Path object to an absolute Path.

        Args:
            path: String or Path object to resolve.

        Returns:
            Absolute Path object.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def checkpoints_dir(self) -> Optional[Path]:
        """Directory for model checkpoints (optional, may be None)."""
        path = self.config.get("paths", {}).get(
            "checkpoints_dir", f"checkpoints/{self.model_name}"
        )
        if path is not None:
            return self._resolve_path(path)
        return None

    @property
    def results_models_dir(self) -> Path:
        """Directory for saved model files within results."""
        return self.results_dir / "models"

    @property
    def plots_dir(self) -> Path:
        """Directory for generated plots and visualizations."""
        return self.results_dir / "plots"

    @property
    def metrics_dir(self) -> Path:
        """Directory for metrics and evaluation results."""
        return self.results_dir / "metrics"

    @property
    def predictions_dir(self) -> Path:
        """Directory for model predictions output."""
        return self.results_dir / "predictions"

    @property
    def tensorboard_dir(self) -> Path:
        """Directory for TensorBoard event files."""
        return self.results_dir / "tensorboard"

    @property
    def grid_search_dir(self) -> Path:
        """Directory for grid search results."""
        return self.results_dir / "grid_search"

    def _create_dirs(self) -> None:
        """Create all standard model directories if they don't exist."""
        directories = [
            self.results_dir,
            self.models_dir,
            self.results_models_dir,
            self.plots_dir,
            self.metrics_dir,
            self.logs_dir,
            self.predictions_dir,
        ]
        for d in directories:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)

    def get_subdir(self, subdir_name: str) -> Path:
        """
        Get a path to a subdirectory by attribute name, creating it if needed.

        Args:
            subdir_name: Name of the subdirectory attribute
                         (e.g., 'models_dir', 'plots_dir', 'metrics_dir').

        Returns:
            Path to the requested subdirectory.
        """
        if hasattr(self, subdir_name):
            return getattr(self, subdir_name)
        path = self.results_dir / subdir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self) -> str:
        return f"ModelPaths({self.model_name}, base_dir={self.results_dir})"


# ============================================================================
# 4. MAIN PATHS CLASS
# ============================================================================

class Paths:
    """
    Centralized project path management.

    Provides structured, lazy-resolved access to all project directories
    as defined in configs/paths.yaml. Supports model-specific paths
    via configs/models.yaml.

    Usage:
        from src.utils.config import paths

        paths.processed_data_dir
        paths.get_model("xgboost_mono").checkpoints_dir
        paths.list_models()
    """

    def __init__(self):
        """Initialize path manager and load configurations."""
        self._raw = _config.load("paths.yaml")
        self._resolved: Dict[str, Path] = {}

        self._models_config = _config.load("models.yaml")
        self._model_instances: Dict[str, ModelPaths] = {}
        self._models_config_cache: Dict[str, Dict[str, Any]] = (
            self._models_config.get("models", {})
        )

        logger.debug(f"Paths initialized. Project root: {PROJECT_ROOT}")

    def _resolve(self, path_str: str) -> Path:
        """
        Convert a path string to an absolute Path.

        If the path is already absolute, it is returned as-is.
        Otherwise, it is resolved relative to the project root.

        Args:
            path_str: Path string to resolve.

        Returns:
            Absolute Path object.
        """
        path = Path(path_str)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _get_dir(self, key: str) -> Path:
        """
        Retrieve a directory path from project_dirs configuration.

        Supports dot-notation for nested keys (e.g., 'data.processed').

        Args:
            key: Configuration key, possibly with dot-notation nesting.

        Returns:
            Resolved absolute Path to the directory.
        """
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

    def get_model(self, model_name: str) -> ModelPaths:
        """
        Get a ModelPaths instance for a specific model.

        Model configurations are loaded from configs/models.yaml.

        Args:
            model_name: Model name as defined in models.yaml
                        (e.g., 'xgboost_mono', 'cnn_mfcc_mono').

        Returns:
            ModelPaths object providing structured access to model directories.

        Raises:
            KeyError: If the model name is not found in models.yaml.

        Example:
            paths.get_model("xgboost_mono").models_dir
            paths.get_model("cnn_mfcc_mono").checkpoints_dir
        """
        if model_name not in self._model_instances:
            if model_name not in self._models_config_cache:
                available = list(self._models_config_cache.keys())
                raise KeyError(
                    f"Model '{model_name}' not found in configs/models.yaml. "
                    f"Available models: {available}"
                )

            model_config = self._models_config_cache[model_name]
            self._model_instances[model_name] = ModelPaths(model_config, model_name)

        return self._model_instances[model_name]

    def list_models(self) -> List[str]:
        """
        Return a list of all available model names.

        Returns:
            List of model names defined in configs/models.yaml.
        """
        return list(self._models_config_cache.keys())

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get the full configuration dictionary for a model.

        Args:
            model_name: Model name as defined in models.yaml.

        Returns:
            Model configuration dictionary.

        Raises:
            KeyError: If the model name is not found.
        """
        if model_name not in self._models_config_cache:
            raise KeyError(f"Model '{model_name}' not found in configuration")
        return self._models_config_cache[model_name]

    @property
    def xgboost(self) -> ModelPaths:
        """Convenience accessor for the XGBoost model paths."""
        return self.get_model("xgboost_mono")

    @property
    def cnn(self) -> ModelPaths:
        """Convenience accessor for the CNN model paths."""
        return self.get_model("cnn_mfcc_mono")

    @property
    def metadata_dir(self) -> Path:
        """Directory containing FMA metadata CSV files (tracks, features, genres)."""
        return Path(self._raw["external_data"]["metadata_dir"])

    @property
    def active_zip(self) -> Path:
        """Path to the currently active FMA ZIP archive."""
        return Path(self._raw["external_data"]["active_zip"])

    @property
    def active_subset(self) -> str:
        """Name of the currently active FMA subset (small/medium/large)."""
        return self._raw["external_data"]["active_subset"]

    def get_fma_zip(self, subset: str) -> Path:
        """
        Get the path to the FMA ZIP archive for a specific subset.

        Args:
            subset: FMA subset name ('small', 'medium', 'large').

        Returns:
            Path to the corresponding ZIP archive.
        """
        return Path(self._raw["external_data"][f"fma_{subset}_zip"])

    @property
    def raw_data_dir(self) -> Path:
        """Directory for raw (unprocessed) data."""
        return self._get_dir("data.raw")

    @property
    def processed_data_dir(self) -> Path:
        """Directory for processed and cached data."""
        return self._get_dir("data.processed")

    @property
    def external_data_dir(self) -> Path:
        """Directory for external data sources."""
        return self._get_dir("data.external")

    @property
    def processors_data_dir(self) -> Path:
        """Directory for saved preprocessor objects (encoders, scalers)."""
        path = self.processed_data_dir / "processors"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for model training checkpoints."""
        return self._get_dir("checkpoints")

    @property
    def results_dir(self) -> Path:
        """Directory for experiment results and outputs."""
        return self._get_dir("results")

    @property
    def models_dir(self) -> Path:
        """Directory for saved model files."""
        return self._get_dir("models")

    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self._get_dir("logs")

    @property
    def configs_dir(self) -> Path:
        """Directory for project configuration files."""
        return PROJECT_ROOT / "configs"

    @property
    def audio_features_dir(self) -> Path:
        """Directory for cached audio features."""
        path = self.processed_data_dir / "audio_features"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def fma_features_dataset_dir(self) -> Path:
        """Directory for FMA-specific feature datasets."""
        path = self.audio_features_dir / "fma"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def mfcc_cache_dir(self) -> Path:
        """Directory for cached MFCC features."""
        path = self.audio_features_dir / "mfcc"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def spectrogram_cache_dir(self) -> Path:
        """Directory for cached spectrogram features."""
        path = self.audio_features_dir / "spectrograms"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def waveform_cache_dir(self) -> Path:
        """Directory for cached waveform data."""
        path = self.audio_features_dir / "waveforms"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def audio_metadata_file(self) -> Path:
        """Path to the audio metadata JSON file."""
        return self.audio_features_dir / "audio_metadata.json"

    @property
    def zip_template(self) -> str:
        """Template string for FMA ZIP internal path structure."""
        return self._raw.get("zip_structure", {}).get(
            "template", "fma_{subset}/{folder}/{track_id}.mp3"
        )

    @property
    def track_id_padding(self) -> int:
        """Number of digits for zero-padded track IDs in ZIP paths."""
        return self._raw.get("zip_structure", {}).get("track_id_padding", 6)

    @property
    def folder_chars(self) -> int:
        """Number of prefix characters used for folder grouping in ZIP."""
        return self._raw.get("zip_structure", {}).get("folder_chars", 3)

    def get_tracks_csv(self) -> Path:
        """Path to the FMA tracks metadata CSV file."""
        return self.metadata_dir / "tracks.csv"

    def get_features_csv(self) -> Path:
        """Path to the FMA features metadata CSV file."""
        return self.metadata_dir / "features.csv"

    def get_genres_csv(self) -> Path:
        """Path to the FMA genres metadata CSV file."""
        return self.metadata_dir / "genres.csv"

    def ensure_dirs(self) -> None:
        """Create all standard project directories if they don't exist."""
        dirs = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.processors_data_dir,
            self.checkpoints_dir,
            self.results_dir,
            self.models_dir,
            self.logs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.debug("All standard directories ensured")

    def clear_cache(self, cache_type: str = "all") -> None:
        """
        Clear cached data directories.

        Args:
            cache_type: Type of cache to clear.
                        Options: 'all', 'mfcc', 'waveforms', 'spectrograms', 'datasets'.
        """
        import shutil

        cache_targets = {
            "mfcc": self.mfcc_cache_dir,
            "waveforms": self.waveform_cache_dir,
            "spectrograms": self.spectrogram_cache_dir,
        }

        if cache_type in cache_targets or cache_type == "all":
            for name, directory in cache_targets.items():
                if cache_type in ["all", name] and directory.exists():
                    shutil.rmtree(directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Cleared {name} cache: {directory}")

        if cache_type in ["all", "datasets"]:
            if self.fma_features_dataset_dir.exists():
                for f in self.fma_features_dataset_dir.glob("*"):
                    if f.is_file() and f.suffix in [".npy", ".json"]:
                        f.unlink()
                logger.info(f"Cleared datasets in: {self.fma_features_dataset_dir}")

    def print_info(self) -> None:
        """
        Print a human-readable summary of all configured paths.

        This is a manual debugging/exploration utility, not intended
        for use in production pipelines.
        """
        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"\n[External Data]")
        print(f"  metadata_dir: {self.metadata_dir}")
        print(f"  active_zip:   {self.active_zip}")
        print(f"  active_subset: {self.active_subset}")
        print(f"\n[ZIP Structure]")
        print(f"  template:     {self.zip_template}")
        print(f"  padding:      {self.track_id_padding}")
        print(f"  folder_chars: {self.folder_chars}")
        print(f"\n[Internal Directories]")
        print(f"  raw_data:     {self.raw_data_dir}")
        print(f"  processed:    {self.processed_data_dir}")
        print(f"  checkpoints:  {self.checkpoints_dir}")
        print(f"  results:      {self.results_dir}")
        print(f"  models:       {self.models_dir}")
        print(f"\n[Models]")
        for model_name in self.list_models():
            model = self.get_model(model_name)
            print(f"  {model_name}: {model.results_dir}")


# ============================================================================
# 5. AUDIO PARAMETERS
# ============================================================================

class AudioParams:
    """
    Audio processing parameters loaded from configs/audio.yaml.

    Provides structured access to sample rate, FFT parameters,
    mel-spectrogram configuration, MFCC settings, and augmentation options.

    Usage:
        from src.utils.config import audio_params

        sr = audio_params.sample_rate
        n_mels = audio_params.n_mels
    """

    def __init__(self):
        """Load audio configuration from audio.yaml."""
        self._raw = _config.load("audio.yaml")
        logger.debug("AudioParams initialized")

    @property
    def sample_rate(self) -> int:
        """Target sample rate for audio processing (Hz)."""
        return self._raw.get("sample_rate", 22050)

    @property
    def duration(self) -> int:
        """Target audio duration in seconds."""
        return self._raw.get("duration", 30)

    @property
    def n_fft(self) -> int:
        """FFT window size."""
        return self._raw.get("n_fft", 2048)

    @property
    def hop_length(self) -> int:
        """Hop length for STFT (samples between frames)."""
        return self._raw.get("hop_length", 512)

    @property
    def win_length(self) -> int:
        """Window length for STFT."""
        return self._raw.get("win_length", 2048)

    @property
    def n_mels(self) -> int:
        """Number of mel filterbanks."""
        return self._raw.get("n_mels", 128)

    @property
    def fmin(self) -> int:
        """Minimum frequency for mel scale (Hz)."""
        return self._raw.get("fmin", 0)

    @property
    def fmax(self) -> int:
        """Maximum frequency for mel scale (Hz)."""
        return self._raw.get("fmax", 8000)

    @property
    def n_mfcc(self) -> int:
        """Number of MFCC coefficients to extract."""
        return self._raw.get("n_mfcc", 20)

    @property
    def n_chroma(self) -> int:
        """Number of chroma bins (pitch classes)."""
        return self._raw.get("n_chroma", 12)

    @property
    def n_bands(self) -> int:
        """Number of frequency bands for spectral features."""
        return self._raw.get("n_bands", 7)

    @property
    def augmentation(self) -> dict:
        """
        Audio augmentation configuration.

        Returns:
            Dictionary with augmentation parameters
            (e.g., pitch_shift, time_stretch, noise).
        """
        return self._raw.get("augmentation", {})

    def to_dict(self) -> dict:
        """
        Export core audio parameters as a flat dictionary.

        Returns:
            Dictionary with keys: sr, duration, n_fft, hop_length,
            win_length, n_mels, fmin, fmax, n_mfcc.
        """
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

    def print_info(self) -> None:
        """
        Print a human-readable summary of audio parameters.

        This is a manual debugging/exploration utility, not intended
        for use in production pipelines.
        """
        print("[Audio Parameters]")
        print(f"  sample_rate:  {self.sample_rate}")
        print(f"  duration:     {self.duration} sec")
        print(f"  n_mels:       {self.n_mels}")
        print(f"  n_mfcc:       {self.n_mfcc}")
        print(f"  hop_length:   {self.hop_length}")


# ============================================================================
# 6. GLOBAL INSTANCES
# ============================================================================

paths = Paths()
audio_params = AudioParams()


# ============================================================================
# 7. MAIN GUARD
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    setup_logging(level=logging.DEBUG)

    paths.print_info()
    print()
    audio_params.print_info()

    print("\n" + "=" * 50)
    print("MODEL PATH ACCESS TEST")
    print("=" * 50)

    xgb = paths.get_model("xgboost_mono")
    print(f"XGBoost results_dir:    {xgb.results_dir}")
    print(f"XGBoost models_dir:     {xgb.models_dir}")
    print(f"XGBoost plots_dir:      {xgb.plots_dir}")
    print(f"XGBoost checkpoints_dir: {xgb.checkpoints_dir}")

    print(f"\nCNN checkpoints_dir:    {paths.cnn.checkpoints_dir}")
    print(f"\nAvailable models:       {paths.list_models()}")

    print("\nDone.")