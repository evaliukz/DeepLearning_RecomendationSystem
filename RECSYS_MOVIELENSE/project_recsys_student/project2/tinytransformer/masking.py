"""Causal (look-ahead) masking. REFERENCE IMPLEMENTATION."""

import torch


def causal_mask(T, device=None):
    """Lower-triangular boolean keep-mask of shape (T, T). True = KEEP."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
