"""The full TinyTransformer language model. REFERENCE IMPLEMENTATION.

For Project 2 this exact model is reused as a *sequential recommender*: the
"vocabulary" is the set of items (movies), a "sentence" is one user's
chronologically ordered interaction history, and next-token prediction becomes
next-item prediction. Nothing in this file needs to change for Project 2 -- the
recommendation-specific code lives in ``tinytransformer.rec``.
"""

import torch
import torch.nn as nn

from .block import TransformerBlock
from .masking import causal_mask
from .positional import SinusoidalPositionalEncoding


class TinyTransformerLM(nn.Module):
    """A small GPT-style (decoder-only) sequence model."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 block_size=32, d_ff=None, dropout=0.0, pos="sinusoidal", causal=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.causal = causal
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if pos == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(block_size, d_model)
        else:
            raise ValueError(f"unknown pos={pos!r}; expected 'sinusoidal'")
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        """idx: (B, T) integer ids -> logits (B, T, vocab_size)."""
        B, T = idx.shape
        h = self.tok_emb(idx) + self.pos(T).to(idx.device)
        mask = causal_mask(T, device=idx.device) if self.causal else None
        for block in self.blocks:
            h = block(h, mask)
        h = self.ln_f(h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, *, temperature=1.0, top_k=None,
                 top_p=None, generator=None):
        """Autoregressive sampling (Project 1, Part 3). Unused by Project 2."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum - probs > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
