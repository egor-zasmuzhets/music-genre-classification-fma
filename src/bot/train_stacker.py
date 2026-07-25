"""
Обучение XGBoost-стекера поверх CNN + XGBoost.

Вход:  stacking_train.npz / stacking_test.npz из build_stacking_dataset.py
Выход: stacker.json + stacker.meta.json в SAVE_DIR

Запуск:
    python train_stacker.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils.logging_utils import setup_logging

# ═══════════════════════════════════════════════════════════
DATASET_DIR = Path("")
SAVE_DIR    = Path("")

PARAMS = {
    "n_estimators":     1000,      # с early stopping реально используется ~50-80
    "max_depth":        3,         # очень мелкие деревья — стекер, не основная модель
    "learning_rate":    0.015,
    "subsample":        0.6,
    "colsample_bytree": 0.6,
    "reg_alpha":        3.0,       # L1
    "reg_lambda":       5.0,       # L2
    "min_child_weight": 5,        # минимум сэмплов в листе
    "objective":        "multi:softprob",
    "eval_metric":      "mlogloss",
    "tree_method":      "hist",
    "device":           "cuda",    # "cpu" если нет GPU
    "verbosity":        0,
    "random_state":     42,
    "early_stopping_rounds": 30,
}
# ═══════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)


def load_dataset():
    train = np.load(DATASET_DIR / "stacking_train.npz")
    test  = np.load(DATASET_DIR / "stacking_test.npz")
    meta  = json.loads((DATASET_DIR / "meta.json").read_text())
    return train["X"], train["y"], test["X"], test["y"], meta


def evaluate(model, X, y, genre_names, split_name: str) -> dict:
    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)

    acc        = accuracy_score(y, preds)
    f1_macro   = f1_score(y, preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(y, preds, average="weighted", zero_division=0)
    top3 = float(np.mean([y[i] in np.argsort(proba[i])[::-1][:3] for i in range(len(y))]))

    logger.info(
        "[%s]  acc=%.4f  f1_macro=%.4f  f1_weighted=%.4f  top3=%.4f",
        split_name, acc, f1_macro, f1_weighted, top3,
    )

    report = classification_report(
        y, preds, target_names=genre_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc, "f1_macro": f1_macro,
        "f1_weighted": f1_weighted, "top_3_accuracy": top3,
        "classification_report": report,
    }


if __name__ == "__main__":
    setup_logging(level="INFO", mode="console")

    logger.info("Загружаю датасет из %s", DATASET_DIR)
    X_train, y_train, X_test, y_test, meta = load_dataset()
    genre_names = meta["genre_names"]
    n_classes   = meta["n_classes"]

    logger.info(
        "X_train: %s  X_test: %s  классов: %d  признаков: %d",
        X_train.shape, X_test.shape, n_classes, meta["feature_dim"],
    )
    logger.info("Раскладка признаков: %s", meta["feature_layout"])

    model = xgb.XGBClassifier(num_class=n_classes, **PARAMS)

    logger.info("Обучаю стекер (early_stopping=%d)...", PARAMS["early_stopping_rounds"])
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=20,
    )
    logger.info("Лучшая итерация: %d", model.best_iteration)

    train_metrics = evaluate(model, X_train, y_train, genre_names, "train")
    test_metrics  = evaluate(model, X_test,  y_test,  genre_names, "test")

    # сохраняем
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVE_DIR / "stacker.json"
    model.save_model(str(model_path))

    result = {
        "params":       PARAMS,
        "feature_dim":  meta["feature_dim"],
        "feature_layout": meta["feature_layout"],
        "n_classes":    n_classes,
        "genre_names":  genre_names,
        "train":        {k: v for k, v in train_metrics.items() if k != "classification_report"},
        "test":         {k: v for k, v in test_metrics.items()  if k != "classification_report"},
        "test_per_class": test_metrics["classification_report"],
    }
    (SAVE_DIR / "stacker.meta.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

    print(f"\nСтекер сохранён: {model_path.resolve()}")
    print(f"\n  train  acc={train_metrics['accuracy']:.4f}  "
          f"f1_macro={train_metrics['f1_macro']:.4f}")
    print(f"  test   acc={test_metrics['accuracy']:.4f}  "
          f"f1_macro={test_metrics['f1_macro']:.4f}  "
          f"top3={test_metrics['top_3_accuracy']:.4f}")