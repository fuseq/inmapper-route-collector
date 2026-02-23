"""
Anchor Selection Model

For each turn, 3 anchor candidates are scored by a shared MLP.
Softmax over the 3 scores produces a probability distribution.
The model learns which candidate humans prefer.

Each candidate's feature vector has 3 layers:
    A) Per-candidate metrics  (11 dim) — distance, visibility, tier, size, side, ...
    B) Turn context           ( 6 dim) — angle, type, direction, path ratio
    C) Comparative features   ( 6 dim) — rank, gap, is_best flags

Total numeric: 23.  Plus a 3-dim learned embedding for anchor_type.

Architecture:
    Per-candidate shared MLP:  (23 numeric + 3 embedding) = 26 -> 64 -> 32 -> 1
    Then:  softmax(score_1, score_2, score_3)
    Loss:  CrossEntropyLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.features.feature_extractor import NUM_ROOM_TYPES, ANCHOR_NUMERIC_DIM


class AnchorSelectionModel(nn.Module):
    """
    Input:
        numeric_features: (batch, 3, ANCHOR_NUMERIC_DIM)  float
        type_indices:     (batch, 3)                      long
    Output:
        logits: (batch, 3) — raw scores for each candidate
    """

    NUM_CANDIDATES = 3
    NUMERIC_DIM = ANCHOR_NUMERIC_DIM  # 23
    EMBED_DIM = 3

    def __init__(
        self,
        num_room_types: int = NUM_ROOM_TYPES,
        embed_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(num_room_types, embed_dim)

        input_dim = self.NUMERIC_DIM + embed_dim  # 23 + 3 = 26

        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        numeric_features: torch.Tensor,
        type_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            numeric_features: (B, 3, NUMERIC_DIM)
            type_indices:     (B, 3)
        Returns:
            logits: (B, 3)
        """
        B = numeric_features.size(0)

        type_embeds = self.type_embedding(type_indices)  # (B, 3, embed_dim)
        combined = torch.cat([numeric_features, type_embeds], dim=-1)  # (B, 3, 26)

        combined_flat = combined.view(B * self.NUM_CANDIDATES, -1)  # (B*3, 26)
        scores_flat = self.shared_mlp(combined_flat)  # (B*3, 1)
        scores = scores_flat.view(B, self.NUM_CANDIDATES)  # (B, 3)

        return scores

    def predict_proba(
        self,
        numeric_features: torch.Tensor,
        type_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return softmax probabilities (B, 3)."""
        with torch.no_grad():
            logits = self.forward(numeric_features, type_indices)
            return F.softmax(logits, dim=-1)

    def predict(
        self,
        numeric_features: torch.Tensor,
        type_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return index of best candidate per sample (B,)."""
        proba = self.predict_proba(numeric_features, type_indices)
        return proba.argmax(dim=-1)
