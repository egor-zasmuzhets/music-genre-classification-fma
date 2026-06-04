"""
Comprehensive PyTorch Model Analyzer for Audio Genre Classification.

Provides deep analysis with separate visualizations and structured results.
All metrics are returned as a dictionary for easy access.

Typical usage:
    analyzer = PyTorchModelAnalyzer(model, genre_names, device='cuda')
    results = analyzer.full_analysis(test_loader, save_dir='analysis/')

    # Лёгкий доступ к данным:
    accuracy = results["metrics"]["accuracy"]
    f1_macro = results["metrics"]["f1_macro"]
    f1_per_class = results["metrics"]["per_class"]  # {genre: {f1, precision, recall, ...}}
    cm = results["metrics"]["error_analysis"]["confusion_matrix"]
"""

import json
import logging
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, roc_curve, auc, roc_auc_score,
        average_precision_score, matthews_corrcoef, cohen_kappa_score,
        log_loss, brier_score_loss, balanced_accuracy_score,
        zero_one_loss, top_k_accuracy_score, f1_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

logger = logging.getLogger(__name__)


class ConfidenceInterval:
    """Bootstrap confidence intervals."""

    @staticmethod
    def bootstrap_ci(data, statistic=np.mean, n_bootstrap=1000, ci=0.95):
        bootstrap_stats = []
        n = len(data)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            bootstrap_stats.append(statistic(data[indices]))
        bootstrap_stats = np.array(bootstrap_stats)
        lower = np.percentile(bootstrap_stats, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 + ci) / 2 * 100)
        return float(statistic(data)), float(lower), float(upper)


class PyTorchModelAnalyzer:
    """
    Comprehensive PyTorch model analyzer.
    All metrics returned as structured dict, all plots saved separately.
    """

    def __init__(self, model, genre_names, device="cuda", class_weights=None, model_name="AudioCNN"):
        self.model = model
        self.genre_names = genre_names
        self.n_classes = len(genre_names)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.class_weights = class_weights
        self.model_name = model_name

        self.model.to(self.device)
        self.model.eval()

        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.y_pred_logits = None

        self.results: Dict[str, Any] = {
            "metrics": {}, "per_class": {}, "confidence_intervals": {},
            "calibration": {}, "error_analysis": {}, "timing": {},
            "model_info": {
                "name": model_name, "n_classes": self.n_classes,
                "device": str(self.device), "analysis_time": None,
            }
        }

        logger.info("PyTorchModelAnalyzer initialized — classes=%d, device=%s", self.n_classes, self.device)

    @torch.no_grad()
    def collect_predictions(self, dataloader, max_batches=None, progress_bar=True):
        """Run inference and collect predictions. Returns (y_true, y_pred, y_pred_proba)."""
        self.model.eval()
        all_labels, all_preds, all_probas, all_logits = [], [], [], []

        iterator = tqdm(dataloader, desc="Running inference", disable=not progress_bar)
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()

        for batch_idx, batch in enumerate(iterator):
            if max_batches and batch_idx >= max_batches:
                break
            x, y = batch[0], batch[1]
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
            else:
                x = torch.from_numpy(x).float().to(self.device)
            if isinstance(y, torch.Tensor):
                y = y.to(self.device)
            else:
                y = torch.from_numpy(y).long().to(self.device)

            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            all_labels.append(y.cpu().numpy())
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_probas.append(probabilities.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

        if end_time and start_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0
            self.results["timing"]["inference_time_seconds"] = inference_time
            self.results["timing"]["samples_per_second"] = (
                sum(len(lbl) for lbl in all_labels) / inference_time if inference_time > 0 else 0
            )

        self.y_true = np.concatenate(all_labels)
        self.y_pred = np.concatenate(all_preds)
        self.y_pred_proba = np.concatenate(all_probas)
        self.y_pred_logits = np.concatenate(all_logits)

        logger.info("Collected predictions — samples=%d, inference_time=%.2fs",
                    len(self.y_true), self.results.get("timing", {}).get("inference_time_seconds", 0))
        return self.y_true, self.y_pred, self.y_pred_proba

    # ========================================================================
    # METRICS — всё возвращается через self.results["metrics"]
    # ========================================================================

    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics. Returns the full metrics dict.
        Access via:
            results["metrics"]["accuracy"]
            results["metrics"]["f1_macro"]
            results["metrics"]["per_class"]["Rock"]["f1"]
            results["metrics"]["error_analysis"]["confusion_matrix"]
            results["metrics"]["top_k"][3]
        """
        if self.y_true is None:
            raise RuntimeError("Call collect_predictions() first.")

        m = {}  # metrics dict

        # Основные
        m['accuracy'] = float((self.y_pred == self.y_true).mean())

        if HAS_SKLEARN:
            m['balanced_accuracy'] = float(balanced_accuracy_score(self.y_true, self.y_pred))
            m['matthews_corrcoef'] = float(matthews_corrcoef(self.y_true, self.y_pred))
            m['cohen_kappa'] = float(cohen_kappa_score(self.y_true, self.y_pred))
            m['log_loss'] = float(log_loss(self.y_true, self.y_pred_proba))
            m['brier_score'] = None if self.n_classes > 2 else float(brier_score_loss(self.y_true, self.y_pred_proba[:, 1]))
            m['zero_one_loss'] = float(zero_one_loss(self.y_true, self.y_pred))

        # Per-class
        m['per_class'] = self._compute_per_class_detailed_metrics()
        pc = m['per_class']
        m['precision_macro'] = float(np.mean([v['precision'] for v in pc.values()]))
        m['recall_macro'] = float(np.mean([v['recall'] for v in pc.values()]))
        m['f1_macro'] = float(np.mean([v['f1'] for v in pc.values()]))
        m['precision_weighted'] = self._weighted_average('precision', pc)
        m['recall_weighted'] = self._weighted_average('recall', pc)
        m['f1_weighted'] = self._weighted_average('f1', pc)
        m['specificity_macro'] = float(np.mean([v['specificity'] for v in pc.values()]))
        m['npv_macro'] = float(np.mean([v['npv'] for v in pc.values()]))
        m['fpr_macro'] = float(np.mean([v['fpr'] for v in pc.values()]))
        m['fnr_macro'] = float(np.mean([v['fnr'] for v in pc.values()]))

        # Top-k
        m['top_k'] = self._compute_top_k_accuracy()

        # ROC/PR AUC
        if HAS_SKLEARN:
            m['roc_auc'] = self._compute_roc_auc()
            m['pr_auc'] = self._compute_pr_auc()

        # Confidence intervals
        m['confidence_intervals'] = self._compute_confidence_intervals()

        # Calibration
        m['calibration'] = self._compute_calibration_metrics()

        # Error analysis (включает confusion_matrix!)
        m['error_analysis'] = self._compute_error_analysis()

        # Confidence metrics
        m['confidence_metrics'] = self._compute_confidence_metrics()

        # Statistical tests
        m['statistical_tests'] = self._compute_statistical_tests()

        self.results['metrics'] = m
        logger.info("All metrics computed")
        return m

    def _compute_per_class_detailed_metrics(self):
        """Per-class metrics: precision, recall, f1, specificity, npv, fpr, fnr, mcc, tp/fp/fn/tn, support."""
        per_class = {}
        for i in range(self.n_classes):
            tp = int(np.sum((self.y_pred == i) & (self.y_true == i)))
            fp = int(np.sum((self.y_pred == i) & (self.y_true != i)))
            fn = int(np.sum((self.y_pred != i) & (self.y_true == i)))
            tn = int(np.sum((self.y_pred != i) & (self.y_true != i)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            num = (tp * tn) - (fp * fn)
            den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = num / den if den > 0 else 0.0

            per_class[self.genre_names[i]] = {
                'precision': float(precision), 'recall': float(recall),
                'f1': float(f1), 'specificity': float(specificity),
                'npv': float(npv), 'fpr': float(fpr), 'fnr': float(fnr),
                'mcc': float(mcc),
                'informedness': recall + specificity - 1,
                'markedness': precision + npv - 1,
                'support': int(np.sum(self.y_true == i)),
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            }
        return per_class

    def _weighted_average(self, metric_name, per_class):
        total = len(self.y_true)
        if total == 0:
            return 0.0
        return sum(m[metric_name] * m['support'] for m in per_class.values()) / total

    def _compute_top_k_accuracy(self, k_values=None):
        if k_values is None:
            k_values = [1, 3, 5, 10]
        top_k = {}
        for k in k_values:
            if HAS_SKLEARN:
                top_k[k] = float(top_k_accuracy_score(self.y_true, self.y_pred_proba, k=k, labels=range(self.n_classes)))
            else:
                correct = sum(1 for i, t in enumerate(self.y_true) if t in np.argsort(self.y_pred_proba[i])[-k:])
                top_k[k] = correct / len(self.y_true)
        return top_k

    def _compute_roc_auc(self):
        if not HAS_SKLEARN:
            return {}
        results = {
            'macro': float(roc_auc_score(self.y_true, self.y_pred_proba, multi_class='ovr', average='macro')),
            'weighted': float(roc_auc_score(self.y_true, self.y_pred_proba, multi_class='ovr', average='weighted')),
            'per_class': {},
        }
        for i in range(self.n_classes):
            yb = (self.y_true == i).astype(int)
            try:
                results['per_class'][self.genre_names[i]] = float(roc_auc_score(yb, self.y_pred_proba[:, i]))
            except Exception:
                results['per_class'][self.genre_names[i]] = 0.0
        return results

    def _compute_pr_auc(self):
        if not HAS_SKLEARN:
            return {}
        ap_scores = []
        for i in range(self.n_classes):
            yb = (self.y_true == i).astype(int)
            ap_scores.append(average_precision_score(yb, self.y_pred_proba[:, i]))
        results = {'macro': float(np.mean(ap_scores)), 'per_class': {}}
        for i, genre in enumerate(self.genre_names):
            yb = (self.y_true == i).astype(int)
            results['per_class'][genre] = float(average_precision_score(yb, self.y_pred_proba[:, i]))
        return results

    def _compute_confidence_intervals(self):
        cis = {}
        acc_mean, acc_lower, acc_upper = ConfidenceInterval.bootstrap_ci(
            (self.y_pred == self.y_true).astype(float), statistic=np.mean, n_bootstrap=1000, ci=0.95)
        cis['accuracy'] = {'mean': acc_mean, 'lower_95': acc_lower, 'upper_95': acc_upper, 'margin': (acc_upper - acc_lower) / 2}
        return cis

    def _compute_calibration_metrics(self):
        n_bins = 15
        confidences = np.max(self.y_pred_proba, axis=1)
        correct = (self.y_pred == self.y_true).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece, mce, bin_data = 0.0, 0.0, []
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop = in_bin.mean()
            if prop > 0:
                avg_conf = confidences[in_bin].mean()
                avg_acc = correct[in_bin].mean()
                delta = abs(avg_acc - avg_conf)
                ece += prop * delta
                mce = max(mce, delta)
                bin_data.append({'bin': i, 'lower': float(bin_boundaries[i]), 'upper': float(bin_boundaries[i + 1]),
                                 'n_samples': int(in_bin.sum()), 'accuracy': float(avg_acc),
                                 'confidence': float(avg_conf), 'delta': float(delta)})
        return {'ece': float(ece), 'mce': float(mce), 'n_bins': n_bins, 'bin_details': bin_data,
                'max_confidence_gap': float(np.max(np.abs(correct - confidences))), 'reliability_score': 1.0 - ece,
                'brier_score': None}

    def _compute_error_analysis(self):
        """Error analysis + confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred, labels=range(self.n_classes))
        misclassifications = []
        for i in range(self.n_classes):
            row_sum = cm[i, :].sum()
            for j in range(self.n_classes):
                if i != j and cm[i, j] > 0:
                    misclassifications.append({
                        'true': self.genre_names[i], 'pred': self.genre_names[j],
                        'count': int(cm[i, j]), 'rate': float(cm[i, j] / row_sum) if row_sum > 0 else 0.0,
                    })
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        confidences = np.max(self.y_pred_proba, axis=1)
        errors = (self.y_pred != self.y_true)
        return {
            'confusion_matrix': cm.tolist(),
            'top_misclassifications': misclassifications[:20],
            'total_errors': int(errors.sum()),
            'error_rate': float(errors.mean()),
            'error_confidence_distribution': {
                'low_confidence_errors': int(((confidences < 0.5) & errors).sum()),
                'high_confidence_errors': int(((confidences > 0.9) & errors).sum()),
                'mean_error_confidence': float(confidences[errors].mean()) if errors.any() else 0.0,
                'mean_correct_confidence': float(confidences[~errors].mean()) if (~errors).any() else 0.0,
            },
        }

    def _compute_confidence_metrics(self):
        confidences = np.max(self.y_pred_proba, axis=1)
        entropies = -np.sum(self.y_pred_proba * np.log(self.y_pred_proba + 1e-10), axis=1)
        correct_mask = (self.y_pred == self.y_true)
        return {
            'mean_confidence': float(confidences.mean()),
            'std_confidence': float(confidences.std()),
            'mean_confidence_correct': float(confidences[correct_mask].mean()) if correct_mask.any() else 0.0,
            'mean_confidence_wrong': float(confidences[~correct_mask].mean()) if (~correct_mask).any() else 0.0,
            'mean_entropy': float(entropies.mean()),
            'mean_entropy_correct': float(entropies[correct_mask].mean()) if correct_mask.any() else 0.0,
            'mean_entropy_wrong': float(entropies[~correct_mask].mean()) if (~correct_mask).any() else 0.0,
            'confidence_percentiles': {
                '10': float(np.percentile(confidences, 10)), '25': float(np.percentile(confidences, 25)),
                '50': float(np.percentile(confidences, 50)), '75': float(np.percentile(confidences, 75)),
                '90': float(np.percentile(confidences, 90)),
            },
        }

    def _compute_statistical_tests(self):
        tests = {}
        correct = (self.y_pred == self.y_true)
        n_correct = int(correct.sum())
        n_total = len(correct)
        if HAS_SCIPY:
            try:
                result = stats.binomtest(n_correct, n_total, p=1.0 / self.n_classes, alternative='greater')
                p_value = result.pvalue
            except AttributeError:
                try:
                    from scipy.stats import binom_test
                    p_value = binom_test(n_correct, n_total, p=1.0 / self.n_classes, alternative='greater')
                except Exception:
                    p_value = 1.0
            tests['better_than_random'] = {'p_value': float(p_value), 'significant_at_95': bool(p_value < 0.05)}
        else:
            tests['better_than_random'] = {'p_value': None, 'significant_at_95': False}
        return tests

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================

    def plot_confusion_matrix(self, save_path=None, figsize=(14, 12)):
        """Normalized confusion matrix (blue-white)."""
        if not HAS_PLOT:
            return None
        cm = confusion_matrix(self.y_true, self.y_pred)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype('float') / row_sums
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                    xticklabels=self.genre_names, yticklabels=self.genre_names,
                    cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        ax.set_title(f'Confusion Matrix (Normalized) — {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Genre', fontsize=12)
        ax.set_ylabel('True Genre', fontsize=12)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        return fig

    def plot_per_class_f1(self, save_path=None, figsize=(12, 8)):
        """Per-class F1 bar chart, sorted."""
        if not HAS_PLOT or 'per_class' not in self.results.get('metrics', {}):
            return None
        per_class = self.results['metrics']['per_class']
        genres = list(per_class.keys())
        f1_scores = [per_class[g]['f1'] for g in genres]
        supports = [per_class[g]['support'] for g in genres]
        sorted_idx = np.argsort(f1_scores)
        genres_sorted = [genres[i] for i in sorted_idx]
        f1_sorted = [f1_scores[i] for i in sorted_idx]
        supports_sorted = [supports[i] for i in sorted_idx]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(genres)))
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(range(len(genres)), f1_sorted, color=colors, edgecolor='white')
        ax.set_yticks(range(len(genres)))
        ax.set_yticklabels(genres_sorted, fontsize=10)
        ax.set_xlabel('F1 Score', fontsize=12)
        ax.set_title(f'Per-Class F1 Scores — {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        macro_f1 = self.results['metrics'].get('f1_macro', 0)
        ax.axvline(x=macro_f1, color='red', linestyle='--', linewidth=2, label=f'Macro F1: {macro_f1:.3f}')
        for bar, f1, sup in zip(bars, f1_sorted, supports_sorted):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{f1:.3f} (n={sup})', va='center', fontsize=8)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, axis='x')
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        return fig

    def plot_roc_curves(self, save_path=None, figsize=(12, 8)):
        """ROC curves for all classes."""
        if not HAS_PLOT or not HAS_SKLEARN:
            return None
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab20(np.linspace(0, 1, self.n_classes))
        for i in range(self.n_classes):
            yb = (self.y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(yb, self.y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, alpha=0.6, label=f'{self.genre_names[i]} (AUC={roc_auc:.3f})', color=colors[i])
        ax.plot([0, 1], [0, 1], 'k:', lw=2, label='Random')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves — {self.model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        return fig

    def plot_calibration_curve(self, save_path=None, n_bins=15, figsize=(10, 8)):
        """Reliability diagram."""
        if not HAS_PLOT:
            return None
        confidences = np.max(self.y_pred_proba, axis=1)
        correct = (self.y_pred == self.y_true).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_accuracies, bin_confidences, bin_counts = [], [], []
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            bin_counts.append(in_bin.sum())
            if in_bin.any():
                bin_accuracies.append(correct[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
            else:
                bin_accuracies.append(0.0); bin_confidences.append(0.0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
        ax.plot(bin_confidences, bin_accuracies, 'bo-', lw=2, markersize=8, label='Model')
        ece = self.results.get('metrics', {}).get('calibration', {}).get('ece', 0)
        ax2 = ax.twinx()
        ax2.bar(bin_centers, bin_counts, width=1.0/n_bins*0.8, alpha=0.3, color='gray')
        ax2.set_ylabel('Samples', fontsize=10, color='gray')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Reliability Diagram (ECE={ece:.4f})', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        return fig

    def plot_confidence_histogram(self, save_path=None, figsize=(12, 5)):
        """Confidence histogram (correct vs wrong)."""
        if not HAS_PLOT:
            return None
        confidences = np.max(self.y_pred_proba, axis=1)
        correct_mask = (self.y_pred == self.y_true)
        fig, ax = plt.subplots(figsize=figsize)
        bins = np.linspace(0, 1, 30)
        ax.hist(confidences[correct_mask], bins=bins, alpha=0.7, label=f'Correct (n={correct_mask.sum()})', color='green')
        ax.hist(confidences[~correct_mask], bins=bins, alpha=0.7, label=f'Wrong (n={(~correct_mask).sum()})', color='red')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Confidence Distribution — {self.model_name}', fontsize=14, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        return fig

    def plot_top_k_accuracy(self, max_k=10, save_path=None, figsize=(10, 6)):
        """Top-k accuracy progression."""
        if not HAS_PLOT:
            return None
        k_values = list(range(1, max_k + 1))
        accuracies = []
        for k in k_values:
            if HAS_SKLEARN:
                acc = float(top_k_accuracy_score(self.y_true, self.y_pred_proba, k=k, labels=range(self.n_classes)))
            else:
                acc = sum(1 for i, t in enumerate(self.y_true) if t in np.argsort(self.y_pred_proba[i])[-k:]) / len(self.y_true)
            accuracies.append(acc)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, accuracies, 'bo-', lw=2, markersize=8, markerfacecolor='red')
        ax.fill_between(k_values, accuracies, alpha=0.2)
        ax.set_xlabel('k', fontsize=12); ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Top-k Accuracy — {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(k_values); ax.set_ylim([0, 1.05]); ax.grid(True, alpha=0.3)
        for k, acc in zip(k_values, accuracies):
            ax.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        return fig

    def plot_error_heatmap(self, save_path=None, figsize=(12, 10)):
        """Misclassification heatmap (diagonal removed)."""
        if not HAS_PLOT:
            return None
        cm = confusion_matrix(self.y_true, self.y_pred)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype('float') / row_sums
        np.fill_diagonal(cm_norm, 0)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Reds', ax=ax,
                    xticklabels=self.genre_names, yticklabels=self.genre_names,
                    cbar_kws={'label': 'Error Rate'})
        ax.set_title(f'Misclassification Patterns — {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Genre', fontsize=12); ax.set_ylabel('True Genre', fontsize=12)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        return fig

    def plot_all_visualizations(self, save_dir, dpi=200):
        """Save all plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams['figure.dpi'] = dpi
        self.plot_confusion_matrix(save_dir / 'confusion_matrix.png')
        self.plot_per_class_f1(save_dir / 'per_class_f1.png')
        self.plot_roc_curves(save_dir / 'roc_curves.png')
        self.plot_calibration_curve(save_dir / 'calibration_curve.png')
        self.plot_confidence_histogram(save_dir / 'confidence_histogram.png')
        self.plot_top_k_accuracy(save_path=save_dir / 'top_k_accuracy.png')
        self.plot_error_heatmap(save_dir / 'error_heatmap.png')
        logger.info("All visualizations saved to: %s", save_dir)

    # ========================================================================
    # REPORT
    # ========================================================================

    def print_detailed_report(self):
        """Print comprehensive analysis report."""
        m = self.results.get('metrics', {})
        print("\n" + "=" * 100)
        print(f"MODEL ANALYSIS REPORT - {self.model_name}")
        print("=" * 100)
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {self.device}")
        print(f"Number of Classes: {self.n_classes}")
        print(f"Test Samples: {len(self.y_true) if self.y_true is not None else 0}")
        print()
        print("=" * 100)
        print("OVERALL PERFORMANCE")
        print("=" * 100)
        print(f"Accuracy:              {m.get('accuracy', 0):.4f}")
        print(f"Balanced Accuracy:     {m.get('balanced_accuracy', 0):.4f}")
        print(f"F1 Score (Macro):      {m.get('f1_macro', 0):.4f}")
        print(f"F1 Score (Weighted):   {m.get('f1_weighted', 0):.4f}")
        print(f"Matthews Corr Coef:    {m.get('matthews_corrcoef', 0):.4f}")
        print(f"Cohen's Kappa:         {m.get('cohen_kappa', 0):.4f}")

        top_k = m.get('top_k', {})
        if top_k:
            print("\n" + "-" * 50)
            print("TOP-K ACCURACY")
            for k, acc in sorted(top_k.items()):
                print(f"  Top-{k}: {acc:.4f}")

        calibration = m.get('calibration', {})
        if calibration:
            print(f"\nCALIBRATION: ECE={calibration.get('ece', 0):.4f}, MCE={calibration.get('mce', 0):.4f}, Reliability={calibration.get('reliability_score', 0):.4f}")

        roc_auc = m.get('roc_auc', {})
        if roc_auc:
            print(f"\nROC-AUC: Macro={roc_auc.get('macro', 0):.4f}, Weighted={roc_auc.get('weighted', 0):.4f}")

        ea = m.get('error_analysis', {})
        if ea:
            print(f"\nERROR ANALYSIS: {ea.get('total_errors', 0)} errors ({ea.get('error_rate', 0):.2%})")
            top_mis = ea.get('top_misclassifications', [])[:5]
            if top_mis:
                print("  Top-5 misclassifications:")
                for mis in top_mis:
                    print(f"    {mis['true']} → {mis['pred']}: {mis['count']}× ({mis['rate']:.1%})")

        cm = m.get('confidence_metrics', {})
        if cm:
            print(f"\nCONFIDENCE: Correct={cm.get('mean_confidence_correct', 0):.4f}, Wrong={cm.get('mean_confidence_wrong', 0):.4f}, Gap={cm.get('mean_confidence_correct', 0) - cm.get('mean_confidence_wrong', 0):.4f}")

        per_class = m.get('per_class', {})
        if per_class:
            print("\n" + "=" * 100)
            print("PER-CLASS PERFORMANCE")
            print("=" * 100)
            print(f"{'Genre':<20} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Supp':<8}")
            print("-" * 100)
            for genre, pc in per_class.items():
                print(f"{genre:<20} {pc['precision']:.4f}  {pc['recall']:.4f}  {pc['f1']:.4f}  {pc['support']:<8}")
            best = max(per_class.items(), key=lambda x: x[1]['f1'])
            worst = min(per_class.items(), key=lambda x: x[1]['f1'])
            print(f"\nBest:  {best[0]} (F1={best[1]['f1']:.4f})")
            print(f"Worst: {worst[0]} (F1={worst[1]['f1']:.4f})")
        print("\n" + "=" * 100 + "\n")

    def export_results(self, save_path):
        """Export full results to JSON."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        export_data = {
            "model_info": self.results.get("model_info", {}),
            "metrics": self.results.get("metrics", {}),
            "timing": self.results.get("timing", {}),
            "analysis_timestamp": datetime.now().isoformat(),
        }
        def convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [convert(item) for item in obj]
            return obj
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(convert(export_data), f, indent=2, ensure_ascii=False)
        logger.info("Results exported: %s", save_path)

    def full_analysis(self, dataloader, save_dir=None, generate_plots=True, export_json=True, max_batches=None) -> Dict[str, Any]:
        """
        Run complete analysis. Returns full results dict.

        Access:
            results["metrics"]["accuracy"]
            results["metrics"]["per_class"]["Rock"]["f1"]
            results["metrics"]["error_analysis"]["confusion_matrix"]
        """
        analysis_start = datetime.now()
        self.collect_predictions(dataloader, max_batches=max_batches)
        self.compute_all_metrics()
        self.results["model_info"]["analysis_time"] = analysis_start.isoformat()
        self.results["model_info"]["n_samples"] = len(self.y_true) if self.y_true is not None else 0

        if generate_plots and save_dir:
            self.plot_all_visualizations(save_dir / "plots")
        if export_json and save_dir:
            self.export_results(save_dir / "analysis_results.json")
        self.print_detailed_report()

        if save_dir:
            report_path = save_dir / "analysis_report.txt"
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            self.print_detailed_report()
            report_content = sys.stdout.getvalue()
            sys.stdout = old_stdout
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
        return self.results


def analyze_pytorch_model(model, dataloader, genre_names, device="cuda", save_dir=None, model_name="PyTorchModel") -> Dict[str, Any]:
    """Convenience function. Returns full results dict."""
    analyzer = PyTorchModelAnalyzer(model=model, genre_names=genre_names, device=device, model_name=model_name)
    return analyzer.full_analysis(dataloader=dataloader, save_dir=save_dir, generate_plots=True, export_json=True)