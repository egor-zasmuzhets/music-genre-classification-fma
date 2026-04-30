#!/usr/bin/env python
"""
src/training/train_xgboost.py
Обучение XGBoost модели с комплексной оценкой (MONO Classification)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_processed import load_data
from src.data.config import paths
from src.models.xgboost_model import XGBoostGenreClassifier
from src.training.analyzer import ModelAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model for mono classification')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--subset', type=str, default='medium', help='FMA subset')
    parser.add_argument('--min_samples', type=int, default=100, help='Min samples per genre')
    parser.add_argument('--no_weights', action='store_true', help='Disable class weights')
    parser.add_argument('--grid_search', action='store_true', help='Run grid search')
    parser.add_argument('--grid_size', type=str, default='small',
                        choices=['small', 'medium', 'full'], help='Grid search size')
    args = parser.parse_args()

    print("=" * 70)
    print("ОБУЧЕНИЕ XGBOOST МОДЕЛИ (MONO CLASSIFICATION)")
    print("Single-Label Genre Classification — предсказание главного жанра")
    print("=" * 70)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Подмножество: {args.subset}")
    print(f"Min samples: {args.min_samples}")
    print(f"Веса классов: {not args.no_weights}")
    print(f"Grid Search: {args.grid_search}")
    if args.grid_search:
        print(f"Grid size: {args.grid_size}")
    print("-" * 70)

    # 1. Загружаем данные
    print("\n[1/5] Загрузка данных...")
    data = load_data(
        subset=args.subset,
        min_samples_per_genre=args.min_samples
    )

    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    genre_names = data['genre_names']

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Жанров: {len(genre_names)}")

    # Распределение жанров
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n  Распределение жанров в train:")
    for genre_id, count in zip(unique, counts):
        print(f"    {genre_names[genre_id]:20s}: {count} ({count / len(y_train) * 100:.1f}%)")

    # 2. Обучение или Grid Search
    if args.grid_search:
        print("\n[2/5] Запуск Grid Search...")
        from src.training.grid_search import XGBoostGridSearch, GRID_SMALL, GRID_MEDIUM, GRID_FULL

        if args.grid_size == 'small':
            param_grid = GRID_SMALL
        elif args.grid_size == 'medium':
            param_grid = GRID_MEDIUM
        else:
            param_grid = GRID_FULL

        grid_search = XGBoostGridSearch(
            param_grid=param_grid,
            use_class_weights=not args.no_weights,
            verbose=True
        )

        results = grid_search.fit(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            genre_names
        )

        grid_search.save_results()

        best_params = grid_search.get_best_params(metric='composite_score')

        print("\n[Сравнение] Лучшие модели по разным метрикам:")
        best_by_metric = grid_search.get_best_by_metric()
        for metric, info in best_by_metric.items():
            print(f"  {metric}: composite={info['metrics']['composite_score']:.4f}, "
                  f"f1_macro={info['metrics']['f1_macro']:.4f}, "
                  f"top3={info['metrics']['top_3_acc']:.4f}")

        print("\n[3/5] Обучение финальной модели с лучшими параметрами...")
        model = XGBoostGenreClassifier(
            params=best_params,
            use_class_weights=not args.no_weights
        )
    else:
        print("\n[2/5] Обучение модели...")
        model = XGBoostGenreClassifier(
            config_path=args.config,
            use_class_weights=not args.no_weights
        )

    # Обучаем финальную модель
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        genre_names=genre_names,
        verbose=True
    )

    # 4. Комплексная оценка
    print("\n[4/5] Комплексная оценка модели...")
    metrics = model.comprehensive_evaluate(X_test, y_test, genre_names)

    print(f"\n{'=' * 70}")
    print("РЕЗУЛЬТАТЫ КОМПЛЕКСНОЙ ОЦЕНКИ (MONO CLASSIFICATION)")
    print(f"{'=' * 70}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"F1-macro:          {metrics['f1_macro']:.4f}")
    print(f"F1-weighted:       {metrics['f1_weighted']:.4f}")
    print(f"F1-micro:          {metrics['f1_micro']:.4f}")
    print(f"\nTop-1 Accuracy:    {metrics['top_1_accuracy']:.4f}")
    print(f"Top-3 Accuracy:    {metrics['top_3_accuracy']:.4f}")
    print(f"Top-5 Accuracy:    {metrics['top_5_accuracy']:.4f}")
    print(f"\nROC-AUC (ovo):     {metrics['roc_auc_ovo']:.4f}")
    print(f"ROC-AUC (ovr):     {metrics['roc_auc_ovr']:.4f}")
    print(f"\nComposite Score:   {metrics['composite_score']:.4f}")
    print(f"\nАнализ уверенности:")
    conf = metrics['confidence']
    print(f"  Уверенность (прав.): {conf['confidence_correct_mean']:.3f} ± {conf['confidence_correct_std']:.3f}")
    print(f"  Уверенность (ошиб.): {conf['confidence_wrong_mean']:.3f} ± {conf['confidence_wrong_std']:.3f}")
    print(f"  Разрыв:               {conf['confidence_gap']:.3f}")
    print(f"  Низкая уверенность:   {conf['low_confidence_rate']:.2%}")
    print(f"{'=' * 70}")

    # 5. Сохраняем модель и результаты
    print("\n[5/5] Сохранение...")
    model.save()

    # Сохраняем метрики в JSON
    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'task': 'mono_classification',
        'task_description': 'Single-label genre classification (main genre only)',
        'model_type': 'xgboost',
        'subset': args.subset,
        'min_samples_per_genre': args.min_samples,
        'use_class_weights': not args.no_weights,
        'params': model.params,
        'metrics': {k: v for k, v in metrics.items() if k not in ['classification_report', 'confidence']},
        'confidence_analysis': metrics['confidence'],
        'genre_names': genre_names,
        'class_metrics': metrics.get('classification_report')
    }

    metrics_path = paths.xgboost_mono_metrics_dir / "comprehensive_results.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Модель обучена и сохранена!")
    print(f"   Модель: {paths.xgboost_mono_models_dir / 'xgboost_best.json'}")
    print(f"   Результаты: {metrics_path}")
    print(f"   Графики: {paths.xgboost_mono_plots_dir}")

    # Визуализация
    print("\n[Визуализация] Генерация отчётов...")
    analyzer = ModelAnalyzer(model, genre_names)
    analysis = analyzer.analyze_predictions(X_test, y_test)
    analyzer.print_analysis_report(analysis)

    return metrics


if __name__ == "__main__":
    main()