"""
PyTorch Dataset classes for StepFilter, AnchorSelection,
and DescriptionStrategy models.

Handles:
  - Converting route records into tensors
  - Stratified train/validation splitting
  - Optional feature normalisation via running statistics
"""
import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.features.feature_extractor import (
    FeatureExtractor, NUM_ROOM_TYPES,
    ANCHOR_TYPE_IDX_COL, ANCHOR_NUMERIC_DIM, ANCHOR_RAW_DIM,
    STRATEGY_FEATURE_DIM,
)
from ml.data.data_loader import CONF_UNMATCHED, CONF_SKIPPED


# ─────────────────────────────────────────────
#  Step Filter Dataset
# ─────────────────────────────────────────────

class StepFilterDataset(Dataset):
    """
    Each sample is a single step from a route.
    X = 20-dim feature vector
    y = 0 (delete) or 1 (keep)
    """

    def __init__(
        self,
        route_records: List[Dict],
        feature_extractor: FeatureExtractor = None,
    ):
        self.fe = feature_extractor or FeatureExtractor()
        self.features: List[np.ndarray] = []
        self.labels: List[int] = []
        self.meta: List[Dict] = []

        for rec in route_records:
            step_feats = self.fe.extract_step_features(rec)
            step_labels = rec.get('step_labels', {})
            steps = rec.get('steps', [])

            for i, step in enumerate(steps):
                sn = step.get('step_number', i + 1)
                if sn not in step_labels:
                    continue
                self.features.append(step_feats[i])
                self.labels.append(step_labels[sn])
                self.meta.append({
                    'route_id': rec.get('route_id', ''),
                    'step_number': sn,
                    'action': step.get('action', ''),
                })

        self.features = np.array(self.features, dtype=np.float32) if self.features else np.zeros((0, 15), dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

    def label_distribution(self) -> Dict[str, int]:
        if len(self.labels) == 0:
            return {'keep': 0, 'delete': 0}
        keep = int(self.labels.sum())
        return {'keep': keep, 'delete': len(self.labels) - keep}


# ─────────────────────────────────────────────
#  Anchor Selection Dataset
# ─────────────────────────────────────────────

class AnchorSelectionDataset(Dataset):
    """
    Each sample is a turn with 3 anchor candidates.
    X = (3, ANCHOR_NUMERIC_DIM) feature matrix  (3 candidates x 23 numeric features)
    type_indices = (3,) integer indices for anchor_type embedding
    y = index of selected candidate (0, 1, or 2)
    
    The anchor_type field is at column ANCHOR_TYPE_IDX_COL (last column)
    in the raw feature vector. It is split out for the Embedding layer.

    When exclude_unmatched=True (default), samples whose anchor label has
    confidence 'unmatched' are dropped -- the human wrote free text that
    referenced an anchor outside the 3 candidates, so the default-to-primary
    label would be noise.
    """

    def __init__(
        self,
        route_records: List[Dict],
        feature_extractor: FeatureExtractor = None,
        exclude_unmatched: bool = True,
    ):
        self.fe = feature_extractor or FeatureExtractor()
        self.features: List[np.ndarray] = []
        self.type_indices: List[np.ndarray] = []
        self.labels: List[int] = []
        self.meta: List[Dict] = []
        self._conf_counts = {
            'confirmed': 0, 'matched': 0, 'unmatched': 0, 'skipped': 0,
        }

        for rec in route_records:
            anchor_records = self.fe.extract_anchor_features(rec)
            anchor_labels = rec.get('anchor_labels', {})
            anchor_confidence = rec.get('anchor_confidence', {})

            for arec in anchor_records:
                sn = arec['step_number']
                if sn not in anchor_labels:
                    continue

                conf = anchor_confidence.get(sn, '')
                self._conf_counts[conf] = self._conf_counts.get(conf, 0) + 1

                if conf == CONF_SKIPPED:
                    continue
                if exclude_unmatched and conf == CONF_UNMATCHED:
                    continue

                feat = arec['features']  # (3, ANCHOR_RAW_DIM)
                type_col = feat[:, ANCHOR_TYPE_IDX_COL].astype(np.int64)
                numeric_feat = np.delete(feat, ANCHOR_TYPE_IDX_COL, axis=1)  # (3, ANCHOR_NUMERIC_DIM)

                self.features.append(numeric_feat.astype(np.float32))
                self.type_indices.append(type_col)
                self.labels.append(anchor_labels[sn])
                self.meta.append({
                    'route_id': rec.get('route_id', ''),
                    'step_number': sn,
                    'candidates': arec['candidates'],
                    'confidence': conf,
                })

        if self.features:
            self.features = np.array(self.features, dtype=np.float32)
            self.type_indices = np.array(self.type_indices, dtype=np.int64)
        else:
            self.features = np.zeros((0, 3, ANCHOR_NUMERIC_DIM), dtype=np.float32)
            self.type_indices = np.zeros((0, 3), dtype=np.int64)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),       # (3, ANCHOR_NUMERIC_DIM)
            torch.from_numpy(self.type_indices[idx]),    # (3,) long
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def label_distribution(self) -> Dict[int, int]:
        dist = {}
        for lbl in self.labels:
            dist[int(lbl)] = dist.get(int(lbl), 0) + 1
        return dist

    def confidence_distribution(self) -> Dict[str, int]:
        """Return counts of each anchor confidence tag seen during loading."""
        return dict(self._conf_counts)


# ─────────────────────────────────────────────
#  Description Strategy Dataset
# ─────────────────────────────────────────────

class DescriptionStrategyDataset(Dataset):
    """
    Each sample is a KEPT step from a route.
    X = 16-dim strategy feature vector
    y = 1 (ANCHOR_BASED) or 0 (NON_ANCHOR)

    Only steps with step_label == 1 AND a strategy_label entry are included.
    """

    def __init__(
        self,
        route_records: List[Dict],
        feature_extractor: FeatureExtractor = None,
    ):
        self.fe = feature_extractor or FeatureExtractor()
        self.features: List[np.ndarray] = []
        self.labels: List[int] = []
        self.meta: List[Dict] = []

        for rec in route_records:
            strategy_feats = self.fe.extract_strategy_features(rec)
            step_labels = rec.get('step_labels', {})
            strategy_labels = rec.get('strategy_labels', {})
            steps = rec.get('steps', [])

            for i, step in enumerate(steps):
                sn = step.get('step_number', i + 1)
                if step_labels.get(sn, 0) != 1:
                    continue
                if sn not in strategy_labels:
                    continue
                self.features.append(strategy_feats[i])
                self.labels.append(strategy_labels[sn])
                self.meta.append({
                    'route_id': rec.get('route_id', ''),
                    'step_number': sn,
                    'action': step.get('action', ''),
                })

        if self.features:
            self.features = np.array(self.features, dtype=np.float32)
        else:
            self.features = np.zeros((0, STRATEGY_FEATURE_DIM), dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

    def label_distribution(self) -> Dict[str, int]:
        if len(self.labels) == 0:
            return {'anchor_based': 0, 'non_anchor': 0}
        anchor = int(self.labels.sum())
        return {'anchor_based': anchor, 'non_anchor': len(self.labels) - anchor}


# ─────────────────────────────────────────────
#  Train / Validation Split
# ─────────────────────────────────────────────

def train_val_split(
    route_records: List[Dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split route records into train/val sets at the *route* level
    (not step level) to avoid data leakage.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(route_records))
    rng.shuffle(indices)

    split_point = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:split_point]
    val_idx = indices[split_point:]

    train = [route_records[i] for i in train_idx]
    val = [route_records[i] for i in val_idx]

    return train, val


def save_dataset_cache(route_records: List[Dict], path: str):
    """Save pre-processed route records to JSON for faster reloading."""

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    serializable = []
    for rec in route_records:
        clean = {}
        for k, v in rec.items():
            if k in ('turns',):
                cleaned_turns = []
                for t in v:
                    ct = {}
                    for tk, tv in t.items():
                        if tk.startswith('_'):
                            continue
                        if isinstance(tv, tuple):
                            ct[tk] = list(tv)
                        elif isinstance(tv, (list, dict, int, float, str, bool, type(None))):
                            ct[tk] = tv
                        else:
                            ct[tk] = str(tv)
                    cleaned_turns.append(ct)
                clean[k] = cleaned_turns
            elif k in ('path_points',):
                clean[k] = [list(p) if isinstance(p, tuple) else p for p in v]
            elif k in ('start_room', 'end_room'):
                room = {}
                for rk, rv in v.items():
                    room[rk] = list(rv) if isinstance(rv, tuple) else rv
                clean[k] = room
            else:
                clean[k] = v
        serializable.append(clean)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, default=_convert)
    print(f"[Dataset] Cached {len(serializable)} records to {path}")


def load_dataset_cache(path: str) -> List[Dict]:
    """Load cached route records from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for rec in data:
        if 'path_points' in rec:
            rec['path_points'] = [tuple(p) for p in rec['path_points']]
        for turn in rec.get('turns', []):
            if 'point' in turn:
                turn['point'] = tuple(turn['point'])
            if 'anchor' in turn and turn['anchor']:
                a = turn['anchor']
                if isinstance(a, list):
                    if len(a) > 4 and isinstance(a[4], list):
                        a[4] = [tuple(alt) for alt in a[4]]
                    turn['anchor'] = tuple(a)
    print(f"[Dataset] Loaded {len(data)} cached records from {path}")
    return data
