# Auto-Generated Documentation

> **PR:** #2 - XGBoost baseline (mono-classification)
> **Author:** @egor-zasmuzhets
> **Generated:** 2026-05-04 07:56:52 UTC


## 📄 `configs/audio.yaml`

**Description:** This YAML code defines a set of audio processing parameters, including sample rate, duration, and various spectral features such as n_fft, hop_length, and mel-frequency cepstral coefficients (MFCCs). It also specifies augmentation techniques like time stretching, pitch shifting, and noise addition.

**Functions:** None

**Classes:** None

**Dependencies:** librosa, pydub

---

## 📄 `configs/model_config.yaml`

**Description:** This YAML code defines a configuration for training an XGBoost model for music genre classification. It includes model parameters, training settings, selection metrics, and data configuration.

**Functions:** None

**Classes:** None

**Dependencies:** xgboost

---

## 📄 `src/data/config.py`

**Description:** This Python code is used for managing project configurations, including defining the project root, loading YAML configuration files, and providing paths to various directories and files. It includes classes for handling configurations, paths, and audio parameters.

**Functions:** find_project_root, load, reload, ensure_dirs, print_info, to_dict

**Classes:** Config, Paths, AudioParams

**Dependencies:** yaml, pathlib, typing

---

## 📄 `src/models/xgboost_model.py`

**Description:** This code defines a class XGBoostGenreClassifier for music genre classification using XGBoost. It supports training with/without class weights, early stopping on validation, saving/loading models, and comprehensive evaluation metrics.

**Functions:** _get_default_path, _get_sample_weights, fit, predict, predict_proba, top_k_accuracy, confidence_analysis, _compute_composite_score, comprehensive_evaluate, get_feature_importance, save, load, print_summary

**Classes:** XGBoostGenreClassifier

**Dependencies:** json, yaml, numpy, pandas, xgboost, pathlib, sklearn

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

## 📄 `src/training/train_xgboost.py`

**Description:** This script trains an XGBoost model for mono classification. It loads data, trains the model, performs a comprehensive evaluation, and saves the results.

**Functions:** main, load_data, XGBoostGenreClassifier, XGBoostGridSearch, ModelAnalyzer

**Classes:** XGBoostGenreClassifier, XGBoostGridSearch, ModelAnalyzer

**Dependencies:** numpy, argparse, json, pathlib, datetime

---
