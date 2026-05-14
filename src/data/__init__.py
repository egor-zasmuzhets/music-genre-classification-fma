"""
src/data/__init__.py
Data loading, preprocessing, and caching for the FMA dataset.
"""

import logging
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    # These are only for static analysis — never actually imported here
    from src.data.torch_dataset import (
        create_mfcc_dataloaders as _create_mfcc_dataloaders,
        MFCCDataset as _MFCCDataset,
        MFCCAugmentation as _MFCCAugmentation,
    )
    from src.data.mfcc_extractor import (
        MFCCExtractor as _MFCCExtractor,
        MFCCConfig as _MFCCConfig,
    )
    from src.data.audio_loader import AudioLoader as _AudioLoader

__all__ = [
    "FMALoader",
    "DataPreprocessor",
    "DataPipeline",
    "LoadProcessedData",
    "load_data",
    "load_track_indices",
    "list_datasets",
    "create_mfcc_dataloaders",
    "MFCCDataset",
    "MFCCAugmentation",
    "MFCCExtractor",
    "MFCCConfig",
    "AudioLoader",
]

# Lightweight imports — always available
from src.data.loader import FMALoader
from src.data.preprocessor import DataPreprocessor
from src.data.pipeline import DataPipeline
from src.data.load_processed import (
    LoadProcessedData,
    load_data,
    load_track_indices,
    list_datasets,
)

create_mfcc_dataloaders: Any = None
MFCCDataset: Any = None
MFCCAugmentation: Any = None
MFCCExtractor: Any = None
MFCCConfig: Any = None
AudioLoader: Any = None


def __getattr__(name: str) -> Any:
    """Lazy-load heavy dependencies only when accessed."""
    _lazy_map = {
        "create_mfcc_dataloaders": "src.data.torch_dataset",
        "MFCCDataset": "src.data.torch_dataset",
        "MFCCAugmentation": "src.data.torch_dataset",
        "MFCCExtractor": "src.data.mfcc_extractor",
        "MFCCConfig": "src.data.mfcc_extractor",
        "AudioLoader": "src.data.audio_loader",
    }

    if name in _lazy_map:
        module_name = _lazy_map[name]
        module = __import__(module_name, fromlist=[name])
        obj = getattr(module, name)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())