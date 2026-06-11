# Auto-Generated Documentation

> **PR:** #4 - CNN on MFCC data
> **Author:** @egor-zasmuzhets
> **Generated:** 2026-06-11 08:28:16 UTC


## 📄 `configs/audio.yaml`

**Description:** This YAML code defines a set of audio processing parameters, including sample rate, duration, and various spectral features such as n_fft, hop_length, and mel-frequency cepstral coefficients (MFCCs). It also specifies augmentation techniques like time stretching, pitch shifting, and noise addition.

**Functions:** None

**Classes:** None

**Dependencies:** librosa, pydub

---

## 📄 `configs/models.yaml`

**Description:** This YAML code defines the configuration for two machine learning models: xgboost_mono and cnn_mfcc_mono. The xgboost_mono model is an XGBoost classifier for music genre classification, while the cnn_mfcc_mono model is a CNN with MFCC features for the same task. The configuration includes model parameters, training settings, and paths for results, checkpoints, and logs.

**Functions:** None

**Classes:** None

**Dependencies:** xgboost, cnn

---

## 📄 `configs/paths.yaml`

**Description:** This YAML configuration file defines the project structure and settings for a music genre classification project. It specifies the project root directory, external data sources, and the structure of the zip files containing the music tracks. The file also defines the project directories for data, checkpoints, results, models, notebooks, and logs.

**Functions:** None

**Classes:** None

**Dependencies:** yaml

---

## 📄 `presentation_figures/adjusted_analysis/adjusted_results.json`

**Description:** This JSON object contains metrics and data related to the performance of a classification model, including accuracy, F1 score, and confusion matrix. It appears to be the output of a machine learning model evaluation.

**Functions:** None

**Classes:** None

**Dependencies:** machine learning library

---

## 📄 `presentation_figures/adjusted_analysis/final_adjusted_results.json`

**Description:** This JSON object contains data related to the performance of a machine learning model, including accuracy, F1 score, and confusion matrix for different classes.

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `presentation_figures/adjusted_analysis/final_results.json`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `presentation_figures/real_analysis/cnn/analysis_results.json`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `presentation_figures/real_analysis/results.json`

**Description:** This JSON code represents a data structure containing performance metrics for two machine learning models, 'xgb' and 'cnn'. The metrics include accuracy, f1_macro, f1_weighted, top_3_accuracy, composite_score, and per_class_f1 for each model.

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/data/__init__.py`

**Description:** This module provides data loading, preprocessing, and caching functionality for the FMA dataset. It includes classes and functions for loading and processing audio data, creating data loaders, and handling MFCC extraction and augmentation.

**Functions:** __getattr__, load_data, load_track_indices, list_datasets

**Classes:** FMALoader, DataPreprocessor, DataPipeline, LoadProcessedData, MFCCDataset, MFCCAugmentation, MFCCExtractor, MFCCConfig, AudioLoader

**Dependencies:** logging, typing, src.data.torch_dataset, src.data.mfcc_extractor, src.data.audio_loader, src.data.loader, src.data.preprocessor, src.data.pipeline, src.data.load_processed

---

## 📄 `src/data/audio_loader.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/data/load_processed.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/data/loader.py`

**Description:** The provided Python code is a metadata loader for the Free Music Archive (FMA) dataset. It provides lazy-loading access to tracks, features, and genres metadata with subset filtering and official train/val/test split extraction.

**Functions:** get_tracks_by_subset, get_available_splits, get_genre_mapping, print_info

**Classes:** FMALoader

**Dependencies:** pandas, pathlib, logging, src.utils.config

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

## 📄 `src/models/cnn_audio.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/models/cnn_mfcc_debug.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

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

## 📄 `src/training/metrics_tracker.py`

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

## 📄 `src/training/pytorch_analyzer.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/training/train_cnn.py`

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
