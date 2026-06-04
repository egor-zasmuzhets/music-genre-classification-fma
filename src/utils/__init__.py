"""
src/utils/__init__.py
Project-wide utilities — configuration, logging, and common helpers.

Provides:
- paths: centralized project path management from configs/paths.yaml
- audio_params: audio processing parameters from configs/audio.yaml
- setup_logging: one-call logging configuration with multiple output modes
- LoggingMixin: mixin class for easy logger access in any class
- LoggingConfig: full logging configuration object (advanced usage)
"""

import logging

__all__ = [
    "paths",
    "audio_params",
    "setup_logging",
    "get_logging_config",
    "LoggingConfig",
    "LoggingMode",
    "LoggingMixin",
]

from src.utils.config import paths, audio_params
from src.utils.logging_utils import (
    setup_logging,
    get_logging_config,
    LoggingConfig,
    LoggingMode,
    LoggingMixin,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())