"""
Description Strategy Model

Binary classifier that predicts whether a kept navigation step should use
an anchor-based description (1) or a non-anchor description (0) such as
proximity ("hemen yandaki"), orientation ("arkanıza alıp"), or geometry
("köşedeki mağaza").

Architecture:
    MLP  16 -> 64 -> 32 -> 1
    Hidden activations: ReLU
    Output activation: Sigmoid (via BCEWithLogitsLoss during training)
"""
import torch
import torch.nn as nn


class DescriptionStrategyModel(nn.Module):
    """
    Input:  (batch, 16) float tensor of strategy features
    Output: (batch,) anchor probability logits
    """

    FEATURE_DIM = 16

    def __init__(self, input_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (use BCEWithLogitsLoss for training)."""
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return anchor probabilities in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions: 1=ANCHOR_BASED, 0=NON_ANCHOR."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()
