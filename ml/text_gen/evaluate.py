"""
Text Generation Evaluation Metrics

Provides BLEU score, anchor ID accuracy, and description quality metrics
for evaluating generated route descriptions against human references.
"""
import re
from typing import Dict, List, Tuple

_ROOM_ID_RE = re.compile(r'(?:Food|Shop|Other|Medical|Social)\s*-\s*ID-?\d+[A-Z]*', re.IGNORECASE)


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute corpus-level BLEU score using sacrebleu.

    Falls back to a simple token-overlap metric if sacrebleu is not installed.
    """
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return {
            'bleu': bleu.score,
            'bleu_bp': bleu.bp,
            'bleu_precisions': bleu.precisions,
        }
    except ImportError:
        return _simple_bleu(predictions, references)


def _simple_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Token-overlap fallback when sacrebleu is not available."""
    if not predictions:
        return {'bleu': 0.0}

    total_overlap = 0.0
    total_pred_len = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if pred_tokens:
            overlap = len(pred_tokens & ref_tokens) / len(pred_tokens)
            total_overlap += overlap
        total_pred_len += 1

    return {'bleu': (total_overlap / max(total_pred_len, 1)) * 100.0}


def anchor_id_accuracy(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Measure how often the generated text contains the same room IDs
    as the reference.

    Returns:
        {
            'anchor_precision': fraction of pred IDs found in ref,
            'anchor_recall': fraction of ref IDs found in pred,
            'anchor_f1': harmonic mean,
            'exact_match_rate': fraction where ID sets match exactly,
        }
    """
    total_precision = 0.0
    total_recall = 0.0
    exact_matches = 0
    n = 0

    for pred, ref in zip(predictions, references):
        pred_ids = set(_ROOM_ID_RE.findall(pred))
        ref_ids = set(_ROOM_ID_RE.findall(ref))

        if not ref_ids and not pred_ids:
            exact_matches += 1
            total_precision += 1.0
            total_recall += 1.0
            n += 1
            continue

        n += 1
        if pred_ids == ref_ids:
            exact_matches += 1

        if pred_ids:
            total_precision += len(pred_ids & ref_ids) / len(pred_ids)
        elif not ref_ids:
            total_precision += 1.0

        if ref_ids:
            total_recall += len(pred_ids & ref_ids) / len(ref_ids)
        elif not pred_ids:
            total_recall += 1.0

    if n == 0:
        return {'anchor_precision': 0, 'anchor_recall': 0, 'anchor_f1': 0, 'exact_match_rate': 0}

    precision = total_precision / n
    recall = total_recall / n
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'anchor_precision': round(precision, 4),
        'anchor_recall': round(recall, 4),
        'anchor_f1': round(f1, 4),
        'exact_match_rate': round(exact_matches / n, 4),
    }


def description_length_stats(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compare average description lengths."""
    pred_lens = [len(p.split()) for p in predictions]
    ref_lens = [len(r.split()) for r in references]

    avg_pred = sum(pred_lens) / max(len(pred_lens), 1)
    avg_ref = sum(ref_lens) / max(len(ref_lens), 1)

    return {
        'avg_pred_words': round(avg_pred, 1),
        'avg_ref_words': round(avg_ref, 1),
        'length_ratio': round(avg_pred / max(avg_ref, 1e-8), 2),
    }


def evaluate_all(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Run all evaluation metrics and return a combined report."""
    results = {}
    results.update(compute_bleu(predictions, references))
    results.update(anchor_id_accuracy(predictions, references))
    results.update(description_length_stats(predictions, references))
    return results


def format_eval_report(metrics: Dict[str, float]) -> str:
    """Format evaluation metrics into a readable string."""
    lines = ['=== Text Generation Evaluation ===']
    lines.append(f"  BLEU Score:         {metrics.get('bleu', 0):.2f}")
    lines.append(f"  Anchor Precision:   {metrics.get('anchor_precision', 0):.4f}")
    lines.append(f"  Anchor Recall:      {metrics.get('anchor_recall', 0):.4f}")
    lines.append(f"  Anchor F1:          {metrics.get('anchor_f1', 0):.4f}")
    lines.append(f"  ID Exact Match:     {metrics.get('exact_match_rate', 0):.4f}")
    lines.append(f"  Avg Pred Words:     {metrics.get('avg_pred_words', 0):.1f}")
    lines.append(f"  Avg Ref Words:      {metrics.get('avg_ref_words', 0):.1f}")
    lines.append(f"  Length Ratio:        {metrics.get('length_ratio', 0):.2f}")
    return '\n'.join(lines)
