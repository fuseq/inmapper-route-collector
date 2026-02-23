"""
Unified Training Script for StepFilter, AnchorSelection, and
DescriptionStrategy models.

Usage:
    python -m ml.training.train --model step_filter --source local
    python -m ml.training.train --model anchor_selector --source sheets --sheets-url <URL>
    python -m ml.training.train --model description_strategy --source local
    python -m ml.training.train --model all --source local

Supports:
    - Early stopping with patience
    - Best model checkpoint saving
    - Train/val loss and metric logging
    - CPU-only execution
"""
import argparse
import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

from ml.data.data_loader import build_training_dataset
from ml.data.dataset import (
    StepFilterDataset,
    AnchorSelectionDataset,
    DescriptionStrategyDataset,
    train_val_split,
    save_dataset_cache,
    load_dataset_cache,
)
from ml.features.feature_extractor import FeatureExtractor
from ml.models.step_filter import StepFilterModel
from ml.models.anchor_selector import AnchorSelectionModel
from ml.models.description_strategy import DescriptionStrategyModel
from ml.evaluation.metrics import (
    binary_metrics,
    multiclass_accuracy,
    top_k_accuracy,
    per_action_breakdown,
    format_metrics_report,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def ensure_checkpoint_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  Step Filter Training
# ─────────────────────────────────────────────

def train_step_filter(
    train_records: list,
    val_records: list,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    dropout: float = 0.1,
) -> dict:
    """Train the StepFilter model and return metrics."""
    ensure_checkpoint_dir()
    fe = FeatureExtractor()

    train_ds = StepFilterDataset(train_records, fe)
    val_ds = StepFilterDataset(val_records, fe)

    print(f"\n[StepFilter] Train: {len(train_ds)} samples {train_ds.label_distribution()}")
    print(f"[StepFilter] Val:   {len(val_ds)} samples {val_ds.label_distribution()}")

    if len(train_ds) == 0:
        print("[StepFilter] No training data available. Skipping.")
        return {}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = StepFilterModel(dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dist = train_ds.label_distribution()
    pos_weight = torch.tensor(
        [dist['delete'] / max(dist['keep'], 1)]
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            n_train += len(y)

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                n_val += len(y)

                preds = (torch.sigmoid(logits) >= 0.5).long().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy().astype(int))

        val_loss /= max(n_val, 1)
        metrics = binary_metrics(np.array(all_labels), np.array(all_preds))

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **metrics,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, 'step_filter_best.pt')
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    best_path = os.path.join(CHECKPOINT_DIR, 'step_filter_best.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))

    model.eval()
    all_preds = []
    all_labels = []
    all_actions = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            X, y = val_ds[i]
            logit = model(X.unsqueeze(0))
            pred = (torch.sigmoid(logit) >= 0.5).long().item()
            all_preds.append(pred)
            all_labels.append(int(y.item()))
            all_actions.append(val_ds.meta[i]['action'])

    final_metrics = binary_metrics(np.array(all_labels), np.array(all_preds))
    action_bd = per_action_breakdown(
        np.array(all_labels), np.array(all_preds), all_actions
    )

    print("\n" + format_metrics_report(
        step_metrics=final_metrics,
        action_breakdown=action_bd,
    ))

    hist_path = os.path.join(CHECKPOINT_DIR, 'step_filter_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    return {'metrics': final_metrics, 'action_breakdown': action_bd, 'history': history}


# ─────────────────────────────────────────────
#  Anchor Selection Training
# ─────────────────────────────────────────────

def train_anchor_selector(
    train_records: list,
    val_records: list,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    dropout: float = 0.1,
) -> dict:
    """Train the AnchorSelection model and return metrics."""
    ensure_checkpoint_dir()
    fe = FeatureExtractor()

    train_ds = AnchorSelectionDataset(train_records, fe)
    val_ds = AnchorSelectionDataset(val_records, fe)

    print(f"\n[AnchorSelector] Train: {len(train_ds)} samples {train_ds.label_distribution()}")
    print(f"[AnchorSelector] Val:   {len(val_ds)} samples {val_ds.label_distribution()}")

    train_conf = train_ds.confidence_distribution()
    val_conf = val_ds.confidence_distribution()
    print(f"[AnchorSelector] Train confidence: {train_conf}")
    print(f"[AnchorSelector] Val   confidence: {val_conf}")

    if len(train_ds) == 0:
        print("[AnchorSelector] No training data available. Skipping.")
        return {}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AnchorSelectionModel(dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for numeric, types, labels in train_loader:
            optimizer.zero_grad()
            logits = model(numeric, types)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            n_train += len(labels)

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        all_preds = []
        all_labels = []
        all_proba = []

        with torch.no_grad():
            for numeric, types, labels in val_loader:
                logits = model(numeric, types)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                n_val += len(labels)

                proba = torch.softmax(logits, dim=-1).numpy()
                preds = logits.argmax(dim=-1).numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_proba.extend(proba)

        val_loss /= max(n_val, 1)
        top1 = multiclass_accuracy(np.array(all_labels), np.array(all_preds))
        top2 = top_k_accuracy(np.array(all_labels), np.array(all_proba), k=2)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'top1_accuracy': top1,
            'top2_accuracy': top2,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"top1={top1:.3f} top2={top2:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, 'anchor_selector_best.pt')
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    best_path = os.path.join(CHECKPOINT_DIR, 'anchor_selector_best.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))

    model.eval()
    all_preds = []
    all_labels = []
    all_proba = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            numeric, types, label = val_ds[i]
            logits = model(numeric.unsqueeze(0), types.unsqueeze(0))
            proba = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            pred = logits.argmax(dim=-1).item()
            all_preds.append(pred)
            all_labels.append(label.item())
            all_proba.append(proba)

    top1 = multiclass_accuracy(np.array(all_labels), np.array(all_preds))
    top2 = top_k_accuracy(np.array(all_labels), np.array(all_proba), k=2)

    anchor_metrics = {
        'top1_accuracy': top1,
        'top2_accuracy': top2,
    }

    print("\n" + format_metrics_report(anchor_metrics=anchor_metrics))

    hist_path = os.path.join(CHECKPOINT_DIR, 'anchor_selector_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    return {'metrics': anchor_metrics, 'history': history}


# ─────────────────────────────────────────────
#  Description Strategy Training
# ─────────────────────────────────────────────

def train_description_strategy(
    train_records: list,
    val_records: list,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    dropout: float = 0.1,
) -> dict:
    """Train the DescriptionStrategy model and return metrics."""
    ensure_checkpoint_dir()
    fe = FeatureExtractor()

    train_ds = DescriptionStrategyDataset(train_records, fe)
    val_ds = DescriptionStrategyDataset(val_records, fe)

    print(f"\n[DescStrategy] Train: {len(train_ds)} samples {train_ds.label_distribution()}")
    print(f"[DescStrategy] Val:   {len(val_ds)} samples {val_ds.label_distribution()}")

    if len(train_ds) == 0:
        print("[DescStrategy] No training data available. Skipping.")
        return {}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DescriptionStrategyModel(dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dist = train_ds.label_distribution()
    pos_weight = torch.tensor(
        [dist['non_anchor'] / max(dist['anchor_based'], 1)]
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            n_train += len(y)

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                n_val += len(y)

                preds = (torch.sigmoid(logits) >= 0.5).long().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy().astype(int))

        val_loss /= max(n_val, 1)
        metrics = binary_metrics(np.array(all_labels), np.array(all_preds))

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **metrics,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, 'description_strategy_best.pt')
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    best_path = os.path.join(CHECKPOINT_DIR, 'description_strategy_best.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))

    model.eval()
    all_preds = []
    all_labels = []
    all_actions = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            X, y = val_ds[i]
            logit = model(X.unsqueeze(0))
            pred = (torch.sigmoid(logit) >= 0.5).long().item()
            all_preds.append(pred)
            all_labels.append(int(y.item()))
            all_actions.append(val_ds.meta[i]['action'])

    final_metrics = binary_metrics(np.array(all_labels), np.array(all_preds))
    action_bd = per_action_breakdown(
        np.array(all_labels), np.array(all_preds), all_actions
    )

    print("\n[DescStrategy] Final validation metrics:")
    print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"  Precision: {final_metrics['precision']:.3f}")
    print(f"  Recall: {final_metrics['recall']:.3f}")
    print(f"  F1: {final_metrics['f1']:.3f}")
    if action_bd:
        print("  Per-action breakdown:")
        for action, bd in action_bd.items():
            print(f"    {action}: {bd}")

    hist_path = os.path.join(CHECKPOINT_DIR, 'description_strategy_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    return {'metrics': final_metrics, 'action_breakdown': action_bd, 'history': history}


# ─────────────────────────────────────────────
#  Annotation Stats
# ─────────────────────────────────────────────

def _log_annotation_stats(records: list):
    """Print a summary of annotation quality across all route records."""
    step_keep = step_del = 0
    conf_counts: dict = {}
    strat_anchor = strat_no_anchor = 0

    for rec in records:
        for lbl in rec.get('step_labels', {}).values():
            if lbl == 1:
                step_keep += 1
            else:
                step_del += 1
        for conf in rec.get('anchor_confidence', {}).values():
            conf_counts[conf] = conf_counts.get(conf, 0) + 1
        for sl in rec.get('strategy_labels', {}).values():
            if sl == 1:
                strat_anchor += 1
            else:
                strat_no_anchor += 1

    total_steps = step_keep + step_del
    print(f"\n[Train] Annotation overview ({len(records)} routes, {total_steps} steps):")
    print(f"  Step labels   -> keep: {step_keep}, delete: {step_del}")
    print(f"  Anchor confidence:")
    for tag in ('confirmed', 'matched', 'unmatched', 'skipped'):
        print(f"    {tag:12s}: {conf_counts.get(tag, 0)}")
    print(f"  Strategy labels -> anchor_based: {strat_anchor}, non_anchor: {strat_no_anchor}")
    print()


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train ML decision models')
    parser.add_argument(
        '--model',
        choices=['step_filter', 'anchor_selector', 'description_strategy', 'both', 'all'],
        default='all', help='Which model(s) to train (both=step_filter+anchor_selector, all=all three)',
    )
    parser.add_argument(
        '--source', choices=['local', 'sheets'],
        default='local', help='Data source',
    )
    parser.add_argument('--sheets-url', type=str, default='', help='Google Sheets Apps Script URL')
    parser.add_argument('--cache', type=str, default='', help='Path to cached dataset JSON')
    parser.add_argument('--venue', type=str, default='zorlu')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.cache and os.path.exists(args.cache):
        print(f"[Train] Loading cached dataset from {args.cache}")
        records = load_dataset_cache(args.cache)
    else:
        records = build_training_dataset(
            source=args.source,
            sheets_url=args.sheets_url,
            venue=args.venue,
            base_dir=PROJECT_ROOT,
        )
        if args.cache:
            save_dataset_cache(records, args.cache)

    if not records:
        print("[Train] No training data available. Exiting.")
        return

    _log_annotation_stats(records)

    train_recs, val_recs = train_val_split(records, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[Train] Split: {len(train_recs)} train / {len(val_recs)} val routes")

    results = {}

    if args.model in ('step_filter', 'both', 'all'):
        print("\n" + "=" * 60)
        print("  TRAINING STEP FILTER MODEL")
        print("=" * 60)
        results['step_filter'] = train_step_filter(
            train_recs, val_recs,
            lr=args.lr, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience,
            dropout=args.dropout,
        )

    if args.model in ('anchor_selector', 'both', 'all'):
        print("\n" + "=" * 60)
        print("  TRAINING ANCHOR SELECTION MODEL")
        print("=" * 60)
        results['anchor_selector'] = train_anchor_selector(
            train_recs, val_recs,
            lr=args.lr, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience,
            dropout=args.dropout,
        )

    if args.model in ('description_strategy', 'all'):
        print("\n" + "=" * 60)
        print("  TRAINING DESCRIPTION STRATEGY MODEL")
        print("=" * 60)
        results['description_strategy'] = train_description_strategy(
            train_recs, val_recs,
            lr=args.lr, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience,
            dropout=args.dropout,
        )

    print("\n[Train] Done. Checkpoints saved to:", CHECKPOINT_DIR)
    return results


if __name__ == '__main__':
    main()
