# Auto-Generated Documentation

> **PR:** #1 - Metadata pipeline
> **Author:** @egor-zasmuzhets
> **Generated:** 2026-04-29 10:13:15 UTC


## 📄 `data/processed/metadata.json`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---

## 📄 `src/data/load_processed.py`

**Description:** This code provides a class LoadProcessedData for loading preprocessed data from a cache directory. It includes methods for checking the existence of data, loading data, and printing information about the loaded data. The code also includes a convenience function load_data for quick import of preprocessed data.

**Functions:** load_data, load

**Classes:** LoadProcessedData

**Dependencies:** pandas, numpy, pickle, json, pathlib, sklearn.preprocessing, joblib

---

## 📄 `src/data/loader.py`

**Description:** This code is designed to load raw data from CSV files of the FMA dataset. It provides a class FMALoader to handle the loading of tracks, features, and genres metadata. The class includes methods to filter tracks by subset and to get available splits for training, validation, and test sets.

**Functions:** get_tracks_by_subset, get_available_splits, get_genre_mapping, print_info

**Classes:** FMALoader

**Dependencies:** pandas, pathlib, typing

---

## 📄 `src/data/pipeline.py`

**Description:** The provided Python code defines a data pipeline for preparing data for classification tasks. It includes loading metadata, filtering by subset, removing tracks without genres, filtering rare genres, obtaining official splits, getting features, encoding labels, normalizing features, and saving the prepared data to a cache.

**Functions:** run, get_class_weights, print_summary, _is_cached, _save_to_cache, _load_from_cache

**Classes:** DataPipeline

**Dependencies:** numpy, pandas, pathlib, typing, pickle, joblib, src.data.config, src.data.loader, src.data.preprocessor

---

## 📄 `src/data/preprocessor.py`

**Description:** This code is for data preprocessing, including filtering rare genres, encoding labels, normalizing features, and calculating class weights.

**Functions:** filter_rare_genres, encode_labels, normalize_features, get_class_weights, save, load

**Classes:** DataPreprocessor

**Dependencies:** numpy, pandas, sklearn.preprocessing, sklearn.utils.class_weight, joblib, pathlib

---

## 📄 `tests/conftest.py`

**Description:** This code provides Pytest fixtures for testing on real FMA Small data. It includes fixtures for loading FMA Small data, tracks, features, and a data pipeline. The fixtures are used to provide data for tests, with some fixtures depending on others.

**Functions:** None

**Classes:** None

**Dependencies:** pytest, numpy, pandas, pathlib, sys

---

## 📄 `tests/data/test_load_processed.py`

**Description:** This code is a set of tests for the LoadProcessedData class, which is responsible for loading processed data. The tests cover various scenarios, including checking if the data exists, loading the data, and verifying its structure and content.

**Functions:** test_load_processed_exists, test_load_processed_data, test_load_data_function, test_load_data_as_dataframe, test_load_processed_print_info, test_load_processed_file_not_found

**Classes:** TestLoadProcessedData

**Dependencies:** pytest, numpy, pandas

---

## 📄 `tests/data/test_loader.py`

**Description:** This module contains tests for the FMALoader class, which is used to load metadata from the FMA Small dataset. The tests cover initialization, loading of tracks, features, and genres, as well as filtering and mapping of genres.

**Functions:** test_loader_initialization, test_tracks_loading, test_features_loading, test_genres_loading, test_get_tracks_by_subset_small, test_get_tracks_by_subset_medium, test_get_available_splits, test_get_genre_mapping, test_tracks_have_genres, test_tracks_have_splits

**Classes:** TestFMALoader

**Dependencies:** pytest, pandas, src.data.loader

---

## 📄 `tests/data/test_pipeline.py`

**Description:** This code is a set of tests for a DataPipeline class, specifically designed to work with the FMA Small dataset. The tests cover various aspects of the pipeline, including initialization, data processing, caching, and feature extraction.

**Functions:** test_pipeline_initialization, test_pipeline_run_returns_data, test_pipeline_metadata, test_pipeline_caching, test_pipeline_class_weights, test_pipeline_print_summary, test_pipeline_feature_shapes, test_pipeline_label_encoder_consistency

**Classes:** TestDataPipeline

**Dependencies:** pytest, numpy, pathlib, src.data.pipeline

---

## 📄 `tests/data/test_preprocessor.py`

**Description:** Unable to analyze code (API error)

**Functions:** None

**Classes:** None

**Dependencies:** None

---
