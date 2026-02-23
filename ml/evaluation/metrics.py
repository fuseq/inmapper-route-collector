"""
Evaluation Metrics for StepFilter and AnchorSelection models.

StepFilter  (binary):  Accuracy, F1, Precision, Recall, Confusion Matrix
AnchorSelection (3-way): Top-1 Accuracy, Top-2 Accuracy, Per-class Accuracy
"""
import numpy as np
from typing import Dict, List, Optional
from collections import Counter


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted labels (0 or 1)
    Returns:
        dict with accuracy, precision, recall, f1
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 2,
) -> np.ndarray:
    """
    Compute NxN confusion matrix.
    cm[i][j] = number of samples with true label i predicted as j.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


def multiclass_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Top-1 accuracy for multi-class classification."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).sum()) / len(y_true)


def top_k_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 2,
) -> float:
    """
    Top-k accuracy: correct if the true label is among the top-k predictions.
    
    Args:
        y_true: (N,) true labels
        y_proba: (N, C) predicted probabilities
        k: number of top predictions to consider
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(y_true) == 0:
        return 0.0

    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)


def per_action_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    actions: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute binary metrics broken down by action type.
    
    Args:
        y_true: (N,) ground truth labels
        y_pred: (N,) predicted labels
        actions: (N,) action type string for each sample
    Returns:
        {action_type: {accuracy, precision, recall, f1}}
    """
    action_groups: Dict[str, List[int]] = {}
    for i, act in enumerate(actions):
        if act not in action_groups:
            action_groups[act] = []
        action_groups[act].append(i)

    results = {}
    for act, indices in sorted(action_groups.items()):
        idx = np.array(indices)
        results[act] = binary_metrics(y_true[idx], y_pred[idx])
        results[act]['count'] = len(indices)

    return results


def format_metrics_report(
    step_metrics: Optional[Dict] = None,
    anchor_metrics: Optional[Dict] = None,
    action_breakdown: Optional[Dict] = None,
) -> str:
    """Format metrics into a human-readable report string."""
    lines = []

    if step_metrics:
        lines.append("=" * 50)
        lines.append("  STEP FILTER MODEL EVALUATION")
        lines.append("=" * 50)
        lines.append(f"  Accuracy:  {step_metrics['accuracy']:.4f}")
        lines.append(f"  Precision: {step_metrics['precision']:.4f}")
        lines.append(f"  Recall:    {step_metrics['recall']:.4f}")
        lines.append(f"  F1 Score:  {step_metrics['f1']:.4f}")
        lines.append(f"  TP={step_metrics['tp']}  FP={step_metrics['fp']}  "
                      f"FN={step_metrics['fn']}  TN={step_metrics['tn']}")
        lines.append("")

    if action_breakdown:
        lines.append("  Per-Action Breakdown:")
        lines.append(f"  {'Action':<14} {'Acc':>6} {'F1':>6} {'P':>6} {'R':>6} {'N':>5}")
        lines.append("  " + "-" * 43)
        for act, m in action_breakdown.items():
            lines.append(
                f"  {act:<14} {m['accuracy']:.3f}  {m['f1']:.3f}  "
                f"{m['precision']:.3f}  {m['recall']:.3f}  {m['count']:>5}"
            )
        lines.append("")

    if anchor_metrics:
        lines.append("=" * 50)
        lines.append("  ANCHOR SELECTION MODEL EVALUATION")
        lines.append("=" * 50)
        lines.append(f"  Top-1 Accuracy: {anchor_metrics['top1_accuracy']:.4f}")
        lines.append(f"  Top-2 Accuracy: {anchor_metrics['top2_accuracy']:.4f}")
        if 'per_class' in anchor_metrics:
            for cls, acc in anchor_metrics['per_class'].items():
                lines.append(f"    Class {cls}: {acc:.4f}")
        lines.append("")

    return "\n".join(lines)
