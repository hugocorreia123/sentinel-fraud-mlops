"""Tabular MLP challenger — small but effective neural net for tabular fraud.

Architecture rationale:
  - Categorical embedding for `type` (5 categories) — neural nets handle
    embeddings more elegantly than one-hot.
  - 3-layer MLP with BatchNorm + Dropout — standard tabular recipe.
  - Output is a single logit (binary classification with BCEWithLogitsLoss).

This is intentionally different from the LightGBM champion: gradient-based
optimization on a smooth decision surface, not greedy axis-aligned splits.
Diversity is the point.
"""
from __future__ import annotations

import torch
from torch import nn


class FraudMLP(nn.Module):
    """A small but effective MLP for tabular fraud classification."""

    def __init__(
        self,
        n_numeric_features: int,
        n_type_categories: int = 5,
        type_emb_dim: int = 4,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(n_type_categories, type_emb_dim)

        layers: list[nn.Module] = []
        in_dim = n_numeric_features + type_emb_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_numeric: torch.Tensor, x_type: torch.Tensor) -> torch.Tensor:
        type_emb = self.type_embedding(x_type)
        x = torch.cat([x_numeric, type_emb], dim=1)
        logits = self.mlp(x).squeeze(-1)
        return logits