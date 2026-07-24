"""Top-k ranking metrics for leave-one-out evaluation.  **PROVIDED.**

Each evaluation example has exactly one held-out relevant item. Given the
model's ranked list of recommended item ids and that one true item, these
score how well the true item was ranked.
"""

import math


def hit_at_k(ranked, true_item, k=10):
    """1.0 if ``true_item`` is in the top-k of ``ranked``, else 0.0."""
    return 1.0 if true_item in ranked[:k] else 0.0


# with a single relevant item, recall@k == hit@k
recall_at_k = hit_at_k


def ndcg_at_k(ranked, true_item, k=10):
    """Normalized discounted cumulative gain with one relevant item."""
    top = list(ranked[:k])
    if true_item in top:
        rank = top.index(true_item)          # 0-based
        return 1.0 / math.log2(rank + 2)
    return 0.0


def mrr(ranked, true_item):
    """Reciprocal rank of the true item (0 if absent)."""
    if true_item in ranked:
        return 1.0 / (ranked.index(true_item) + 1)
    return 0.0
