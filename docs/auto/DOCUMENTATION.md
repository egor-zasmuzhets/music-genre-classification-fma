# Auto-Generated Documentation

> **PR:** #1 - Metadata pipeline
> **Author:** @egor-zasmuzhets
> **Generated:** 2026-04-28 16:55:03 UTC


## 📄 `src/data/load_processed.py`

**Description:** The provided Python code is used for loading preprocessed data from a cache. It includes a class LoadProcessedData that handles the loading of data and provides methods for checking the existence of data, loading the data, and printing information about the loaded data. The code also includes a function load_data that serves as a convenient way to import the data.

**Functions:** load_data, load

**Classes:** LoadProcessedData

**Dependencies:** pandas, numpy, pickle, json, pathlib, sklearn.preprocessing, joblib

---

## 📄 `src/data/loader.py`

**Description:** This module provides a class FMALoader for loading metadata from FMA CSV files. It includes methods for loading tracks, features, and genres, as well as filtering tracks by subset and getting available splits.

**Functions:** get_tracks_by_subset, get_available_splits, get_genre_mapping, print_info

**Classes:** FMALoader

**Dependencies:** pandas, pathlib, typing, src.data.config

---

## 📄 `src/data/pipeline.py`

**Description:** The provided Python code defines a data pipeline for preparing data for classification tasks, specifically designed for the FMA (Free Music Archive) dataset. It includes steps for loading metadata, filtering tracks by subset and genre, encoding labels, normalizing features, and saving the prepared data to a cache.

**Functions:** run, get_class_weights, print_summary, _is_cached, _save_to_cache, _load_from_cache

**Classes:** DataPipeline

**Dependencies:** numpy, pandas, pathlib, typing, pickle, joblib, src.data.config, src.data.loader, src.data.preprocessor

---

## 📄 `src/data/preprocessor.py`

**Description:** This code provides a data preprocessor for classification tasks, specifically for music genres. It filters out rare genres, encodes labels, normalizes features, and calculates class weights.

**Functions:** filter_rare_genres, encode_labels, normalize_features, get_class_weights, save, load

**Classes:** DataPreprocessor

**Dependencies:** numpy, pandas, sklearn.preprocessing, sklearn.utils.class_weight, joblib, pathlib

---
