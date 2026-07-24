"""Sinusoidal positional encoding. REFERENCE IMPLEMENTATION."""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings (Vaswani et al., 2017)."""

    def __init__(self, block_size, d_model):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        pe = torch.zeros(block_size, d_model)
        pos = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, T):
        """Return the first T rows of the table, shape (1, T, d_model)."""
        return self.pe[:T].unsqueeze(0)
