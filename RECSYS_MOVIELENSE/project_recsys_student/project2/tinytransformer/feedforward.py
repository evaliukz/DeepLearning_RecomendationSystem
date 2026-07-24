"""Position-wise feed-forward network. REFERENCE IMPLEMENTATION."""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer MLP applied independently at each position."""

    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)."""
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))
