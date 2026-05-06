# Auto-Generated Documentation

> **PR:** #3 - Data processing pipeline for CNN
> **Author:** @egor-zasmuzhets
> **Generated:** 2026-05-06 09:32:59 UTC


## 📄 `src/data/audio_loader.py`

**Description:** The provided Python code is for an AudioLoader class, which is designed to load audio files from ZIP archives. It supports loading MP3 files directly from ZIP without unpacking, caching loaded tracks, trimming audio to a specified duration, and saving the cache to disk for reuse. The class also provides methods for loading audio with status, loading audio in batches, getting available tracks, and generating reports for failed tracks.

**Functions:** load_audio_with_status, load_audio, load_audio_batch_with_status, get_available_tracks, get_failed_tracks_report

**Classes:** AudioLoader

**Dependencies:** zipfile, io, librosa, numpy, pathlib, hashlib

---

## 📄 `src/data/config.py`

**Description:** This code defines classes and functions for managing paths, configurations, and audio parameters. It provides methods for resolving paths, loading configurations, and accessing various directories and files.

**Functions:** find_project_root, load, reload, _resolve, _get_dir, metadata_dir, active_zip, active_subset, get_fma_zip, raw_data_dir, processed_data_dir, external_data_dir, checkpoints_dir, results_dir, models_dir, logs_dir, configs_dir, xgboost_mono_dir, xgboost_mono_models_dir, xgboost_mono_plots_dir, xgboost_mono_grid_search_dir, xgboost_mono_metrics_dir, xgboost_multi_dir, xgboost_multi_models_dir, xgboost_multi_plots_dir, xgboost_multi_metrics_dir, audio_features_dir, mfcc_cache_dir, spectrogram_cache_dir, waveform_cache_dir, audio_metadata_file, get_tracks_csv, get_features_csv, get_genres_csv, ensure_dirs, print_info, sample_rate, duration, n_fft, hop_length, win_length, n_mels, fmin, fmax, n_mfcc, n_chroma, n_bands, augmentation, to_dict

**Classes:** Config, Paths, AudioParams

**Dependencies:** Optional, Path, Dict, Any

---

## 📄 `src/data/load_processed.py`

**Description:** This code defines a class LoadProcessedData, which appears to be responsible for loading and managing processed data. It includes methods for checking the existence of data, loading data, and printing information.

**Functions:** _get_cache_file, _get_metadata_path, exists, load, load_to_dataframe, print_info, load_track_indices, load_data

**Classes:** LoadProcessedData

**Dependencies:** Path, Dict, Any

---

## 📄 `src/data/mfcc_extractor.py`

**Description:** This code is used for extracting Mel-Frequency Cepstral Coefficients (MFCC) from audio data. It provides a class-based structure for handling MFCC extraction, including configuration, caching, and preparation for CNN input.

**Functions:** extract_from_audio, extract_from_track_id_with_status, extract_from_track_id, get_mfcc_matrix_with_status, get_mfcc_matrix, get_mfcc_with_deltas_with_status, get_mfcc_with_deltas, prepare_for_cnn_with_status, prepare_for_cnn, extract_batch_with_status, get_failed_tracks_report, clear_cache

**Classes:** MFCCConfig, MFCCExtractor

**Dependencies:** numpy, librosa, typing, dataclasses, hashlib, pathlib, pickle

---

## 📄 `src/data/pipeline.py`

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
