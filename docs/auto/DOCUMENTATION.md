# Auto-Generated Documentation

> **PR:** #1 - Metadata pipeline
> **Author:** @egor-zasmuzhets
> **Generated:** 2026-04-29 07:34:33 UTC


## 📄 `src/data/load_processed.py`

**Description:** This code provides functionality for loading preprocessed data from a cache directory. It includes a class LoadProcessedData for handling the data loading process and a function load_data for convenient data loading.

**Functions:** load_data, load

**Classes:** LoadProcessedData

**Dependencies:** pandas, numpy, pickle, json, pathlib, sklearn.preprocessing, joblib

---

## 📄 `src/data/loader.py`

**Description:** This code is responsible for loading raw data from CSV files for the FMA dataset. It provides a class FMALoader to handle the loading of metadata, tracks, features, and genres. The class includes methods to filter tracks by subset and to get available splits for training, validation, and testing.

**Functions:** get_tracks_by_subset, get_available_splits, get_genre_mapping, print_info

**Classes:** FMALoader

**Dependencies:** pandas, pathlib, typing

---

## 📄 `src/data/pipeline.py`

**Description:** The provided code is a data pipeline for preparing data for classification tasks. It includes loading metadata, filtering by subset, removing tracks without genres, filtering rare genres, getting official splits, getting features, encoding labels, normalizing features, and saving the prepared data to a cache.

**Functions:** run, _is_cached, _save_to_cache, _load_from_cache, get_class_weights, print_summary

**Classes:** DataPipeline

**Dependencies:** numpy, pandas, pathlib, typing, pickle, joblib, src.data.config, src.data.loader, src.data.preprocessor

---

## 📄 `src/data/preprocessor.py`

**Description:** This code is designed for data preprocessing, including filtering rare genres, encoding labels, normalizing features, and calculating class weights.

**Functions:** filter_rare_genres, encode_labels, normalize_features, get_class_weights, save, load

**Classes:** DataPreprocessor

**Dependencies:** numpy, pandas, sklearn.preprocessing, sklearn.utils.class_weight, joblib, pathlib

---
