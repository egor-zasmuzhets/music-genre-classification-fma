"""
Строит датасет для обучения XGBoost-стекера поверх CNN + XGBoost.

Схема:
  - train стекера  → val-сплит оригинального датасета (модели не видели его при обучении)
  - val/test стекера → test-сплит оригинального датасета

Для каждого трека:
  - CNN  выдаёт n_classes вероятностей
  - XGBoost выдаёт n_classes вероятностей
  - итоговый вектор: конкатенация = 2 * n_classes признаков (32 при 16 классах)

Результат сохраняется в SAVE_DIR:
  stacking_train.npz   — X_train (N_val,  2*n_classes), y_train
  stacking_test.npz    — X_test  (N_test, 2*n_classes), y_test
  meta.json            — genre_names, n_classes, feature_dim

Запуск:
    python build_stacking_dataset.py
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader

from src.models.cnn_audio import AudioCNN
from src.data.preprocessor import DataPreprocessor
from src.data.load_processed import load_data, load_track_indices
from src.data.torch_dataset import MFCCDataset, MFCCConfig
from src.data.audio_loader import AudioLoader
from src.utils.config import paths
from src.utils.logging_utils import setup_logging

# ═══════════════════════════════════════════════════════════
CNN_CHECKPOINT_PATH = Path("")
XGB_CHECKPOINT_DIR  = Path("")
SAVE_DIR            = Path("")
# ═══════════════════════════════════════════════════════════

_DATASET_ID    = "medium_10"
_SUBSET        = "medium"
_MIN_SAMPLES   = 10
_BATCH_SIZE    = 64
_NUM_WORKERS   = 0

logger = logging.getLogger(__name__)


# ── загрузка моделей ──────────────────────────────────────

def load_cnn(device: str) -> AudioCNN:
    model = AudioCNN.load(CNN_CHECKPOINT_PATH, device=device)
    model.eval()
    logger.info("CNN загружен: %s", CNN_CHECKPOINT_PATH.name)
    return model


def load_xgb(scaler) -> Tuple[xgb.XGBClassifier, object]:
    json_files = sorted(XGB_CHECKPOINT_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"*.json не найден в {XGB_CHECKPOINT_DIR}")
    model = xgb.XGBClassifier()
    model.load_model(str(json_files[0]))
    logger.info("XGBoost загружен: %s", json_files[0].name)
    return model


# ── CNN: прогон DataLoader → вероятности ─────────────────

@torch.no_grad()
def cnn_probas_from_loader(model: AudioCNN, loader: DataLoader,
                           device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает (probas, labels) для всего DataLoader."""
    all_probas, all_labels = [], []

    for x, y in loader:
        if x.dim() == 4 and x.shape[2] == 1:
            x = x.squeeze(2)
        logits = model(x.to(device))
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probas.append(proba)
        all_labels.append(y.numpy())

    return np.concatenate(all_probas), np.concatenate(all_labels)


# ── XGBoost: прогон кешированных признаков → вероятности ─

def xgb_probas_from_cache(xgb_model: xgb.XGBClassifier,
                           scaler, split: str) -> np.ndarray:
    """
    Берёт X_{split}.npy из кеша, нормирует тем же scaler,
    возвращает (N, n_classes) вероятностей.
    split: 'val' или 'test'
    """
    features_path = (paths.fma_features_dataset_dir
                     / _DATASET_ID / "features" / f"X_{split}.npy")
    X = np.load(features_path)
    X_scaled = scaler.transform(X)
    return xgb_model.predict_proba(X_scaled)


# ── сборка одного сплита ──────────────────────────────────

def build_split(split: str,
                cnn_model: AudioCNN,
                xgb_model: xgb.XGBClassifier,
                scaler,
                genre_names: List[str],
                device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Прогоняет split ('val' или 'test') через обе модели.
    Возвращает (X_stacking, y) где X имеет форму (N, 2*n_classes).
    """
    logger.info("── Обрабатываю сплит: %s ──", split)

    # индексы и метки из кеша
    data = load_data(subset=_SUBSET, min_samples_per_genre=_MIN_SAMPLES)
    train_idx, val_idx, test_idx = load_track_indices(
        subset=_SUBSET, min_samples_per_genre=_MIN_SAMPLES)

    indices = val_idx  if split == "val"  else test_idx
    labels  = data[f"y_{split}"]

    # DataLoader для CNN
    mfcc_config = MFCCConfig(include_delta=True, include_delta2=True, n_mfcc=40)
    dataset = MFCCDataset(
        indices=indices, labels=labels,
        audio_loader=AudioLoader(),
        mfcc_config=mfcc_config,
        target_frames=getattr(cnn_model, "target_frames", 430),
        augment=False,
        use_disk_cache=True,
    )
    loader = DataLoader(dataset, batch_size=_BATCH_SIZE,
                        shuffle=False, num_workers=_NUM_WORKERS)

    # CNN вероятности — порядок совпадает с порядком dataset
    cnn_proba, y_cnn = cnn_probas_from_loader(cnn_model, loader, device)
    logger.info("CNN  probas: %s", cnn_proba.shape)

    # XGBoost вероятности — из кеша в том же порядке (indices совпадают)
    xgb_proba = xgb_probas_from_cache(xgb_model, scaler, split)
    logger.info("XGB  probas: %s", xgb_proba.shape)

    assert cnn_proba.shape == xgb_proba.shape, (
        f"Размерности не совпадают: CNN {cnn_proba.shape} vs XGB {xgb_proba.shape}"
    )
    assert len(cnn_proba) == len(labels), (
        f"CNN ({len(cnn_proba)}) и labels ({len(labels)}) не совпадают по длине"
    )

    X_stacking = np.concatenate([cnn_proba, xgb_proba], axis=1)  # (N, 2*n_classes)
    return X_stacking, labels


# ── main ──────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging(level="INFO", mode="console")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Устройство: %s", device)

    # препроцессор — общий источник genre_names и scaler
    prep_dir = paths.fma_features_dataset_dir / _DATASET_ID / "preprocessor"
    preprocessor = DataPreprocessor()
    preprocessor.load(prep_dir)
    genre_names = preprocessor.class_names
    scaler      = preprocessor.scaler
    n_classes   = len(genre_names)
    logger.info("Классов: %d", n_classes)

    cnn_model = load_cnn(device)
    xgb_model = load_xgb(scaler)

    # train стекера = val оригинала
    X_train, y_train = build_split("val",  cnn_model, xgb_model, scaler, genre_names, device)
    # test стекера  = test оригинала
    X_test,  y_test  = build_split("test", cnn_model, xgb_model, scaler, genre_names, device)

    logger.info("X_train: %s  y_train: %s", X_train.shape, y_train.shape)
    logger.info("X_test:  %s  y_test:  %s", X_test.shape,  y_test.shape)

    # сохраняем
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(SAVE_DIR / "stacking_train.npz", X=X_train, y=y_train)
    np.savez(SAVE_DIR / "stacking_test.npz",  X=X_test,  y=y_test)

    meta = {
        "genre_names":   genre_names,
        "n_classes":     n_classes,
        "feature_dim":   X_train.shape[1],
        "feature_layout": f"cnn_proba({n_classes}) | xgb_proba({n_classes})",
        "train_size":    int(len(y_train)),
        "test_size":     int(len(y_test)),
        "dataset_id":    _DATASET_ID,
        "cnn_checkpoint": str(CNN_CHECKPOINT_PATH),
        "xgb_checkpoint": str(XGB_CHECKPOINT_DIR),
    }
    (SAVE_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\nДатасет сохранён в: {SAVE_DIR.resolve()}")
    print(f"  stacking_train.npz  X: {X_train.shape}  y: {y_train.shape}")
    print(f"  stacking_test.npz   X: {X_test.shape}   y: {y_test.shape}")
    print(f"  feature_dim: {X_train.shape[1]}  ({n_classes} CNN + {n_classes} XGB)")