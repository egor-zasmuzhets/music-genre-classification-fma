# Auto-Generated Documentation

> **PR:** #3 - Data processing pipeline for CNN
> **Author:** @egor-zasmuzhets
> **Generated:** 2026-05-14 12:15:40 UTC


## 📄 `configs/models.yaml`

**Description:** This YAML code defines the configuration for two machine learning models: xgboost_mono and cnn_mfcc_mono. The xgboost_mono model is an XGBoost classifier for music genre classification, while the cnn_mfcc_mono model is a CNN with MFCC features for the same task. The configuration includes model parameters, training settings, and file paths.

**Functions:** None

**Classes:** None

**Dependencies:** xgboost, cnn

---

## 📄 `configs/paths.yaml`

**Description:** This YAML code defines the configuration for a music genre classification project. It specifies the project root directory, external data sources, and the structure of the project directories.

**Functions:** None

**Classes:** None

**Dependencies:** yaml

---

## 📄 `src/data/__init__.py`

**Description:** This module provides data loading, preprocessing, and caching functionality for the FMA dataset. It includes classes and functions for loading and processing audio data, as well as utilities for creating data loaders and pipelines.

**Functions:** __getattr__, load_data, load_track_indices, list_datasets

**Classes:** FMALoader, DataPreprocessor, DataPipeline, LoadProcessedData, MFCCDataset, MFCCAugmentation, MFCCExtractor, MFCCConfig, AudioLoader

**Dependencies:** logging, typing, src.data.torch_dataset, src.data.mfcc_extractor, src.data.audio_loader, src.data.loader, src.data.preprocessor, src.data.pipeline, src.data.load_processed

---

## 📄 `src/data/audio_loader.py`

**Description:** This code defines a class called AudioLoader that loads audio files from FMA ZIP archives. It provides features such as lazy ZIP access, multi-level caching, automatic resampling, and detailed status reporting.

**Functions:** load_audio_with_status, load_audio, load_audio_batch_with_status, get_available_tracks, get_failed_tracks_report, close, clear_cache, clear_disk_cache

**Classes:** AudioLoader

**Dependencies:** librosa, numpy, zipfile, pathlib, logging, hashlib, io

---

## 📄 `src/data/load_processed.py`

**Description:** This module provides functions for loading preprocessed data from cache. It supports multiple dataset configurations identified by subset and minimum samples per genre.

**Functions:** load_data, load_track_indices, list_datasets

**Classes:** LoadProcessedData

**Dependencies:** json, logging, pathlib, numpy, pandas, joblib

---

## 📄 `src/data/loader.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/data/mfcc_extractor.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/data/pipeline.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/data/preprocessor.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/data/torch_dataset.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/models/cnn_mfcc_debug.py`

**Description:** This code defines a minimal 2D convolutional neural network (CNN) model for debugging and rapid prototyping of data pipelines, specifically designed for Mel-Frequency Cepstral Coefficients (MFCC) inputs.

**Functions:** forward, save, load, get_num_parameters, print_info

**Classes:** MiniCNN

**Dependencies:** torch, torch.nn, logging, pathlib

---

## 📄 `src/models/xgboost_model.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/training/analyzer.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/training/grid_search.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/training/metrics_tracker_debug.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/training/train_cnn_debug.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/training/train_xgboost.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/utils/__init__.py`

**Description:** This module provides project-wide utilities, including centralized project path management, audio processing parameters, logging configuration, and mixin classes for easy logger access.

**Functions:** setup_logging, get_logging_config

**Classes:** LoggingConfig, LoggingMixin

**Dependencies:** logging, src.utils.config, src.utils.logging_utils

---

## 📄 `src/utils/config.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/utils/logging_utils.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---
