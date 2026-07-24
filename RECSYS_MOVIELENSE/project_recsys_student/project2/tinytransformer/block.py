"""A single pre-norm transformer block. REFERENCE IMPLEMENTATION."""

import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, mask=None):
        """x: (B, T, d_model) -> (B, T, d_model)."""
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x
