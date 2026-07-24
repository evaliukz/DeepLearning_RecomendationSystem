"""Scaled dot-product attention and multi-head attention, in matrix form.

REFERENCE IMPLEMENTATION (Project 1 core, provided as a backup).
See README: you may replace this file with your own Project 1 version.
"""

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None, scale=True):
    """Scaled dot-product attention.

    q: (..., T, d_head)  k: (..., S, d_head)  v: (..., S, d_head)
    mask: optional boolean tensor broadcastable to (..., T, S). True = KEEP.
    Returns out: (..., T, d_head), attn: (..., T, S).
    """
    d_head = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(d_head)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention implemented with explicit head reshaping."""

    def __init__(self, d_model, n_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """(B, T, d_model) -> (B, n_heads, T, d_head)."""
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        """(B, n_heads, T, d_head) -> (B, T, d_model)."""
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * Dh)

    def forward(self, x, mask=None):
        """x: (B, T, d_model) -> (B, T, d_model)."""
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if mask is not None and mask.dim() == 2:
            mask = mask.view(1, 1, *mask.shape)
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        out = self._merge_heads(out)
        return self.dropout(self.out_proj(out))
