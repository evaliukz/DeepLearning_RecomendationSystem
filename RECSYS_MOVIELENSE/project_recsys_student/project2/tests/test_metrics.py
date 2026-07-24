import math
from tinytransformer.rec import metrics


def test_hit():
    assert metrics.hit_at_k([3, 1, 2], 2, k=3) == 1.0
    assert metrics.hit_at_k([3, 1, 2], 9, k=3) == 0.0
    assert metrics.hit_at_k([3, 1, 2], 2, k=2) == 0.0


def test_ndcg():
    assert metrics.ndcg_at_k([5, 1, 2], 5, k=3) == 1.0            # rank 0
    assert abs(metrics.ndcg_at_k([1, 5, 2], 5, k=3) - 1 / math.log2(3)) < 1e-9
    assert metrics.ndcg_at_k([1, 2, 3], 9, k=3) == 0.0


def test_mrr():
    assert metrics.mrr([7, 8, 9], 8) == 0.5
    assert metrics.mrr([7, 8, 9], 1) == 0.0
