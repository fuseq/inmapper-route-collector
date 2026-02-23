"""
Step Filter Model

Binary classifier that predicts whether a navigation step should be
kept (1) or deleted (0) based on numeric route features.

Architecture:
    MLP  20 -> 64 -> 32 -> 1
    Hidden activations: ReLU
    Output activation: Sigmoid
    Loss: BCELoss (or BCEWithLogitsLoss for numerical stability)
"""
import torch
import torch.nn as nn


class StepFilterModel(nn.Module):
    """
    Input:  (batch, 20) float tensor of step features
    Output: (batch, 1) keep probability after sigmoid
    """

    FEATURE_DIM = 20

    def __init__(self, input_dim: int = 20, dropout: float = 0.1):
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
        """Return keep probabilities in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions: 1=keep, 0=delete."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()
