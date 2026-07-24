import torch
from tinytransformer.rec import build_training_sequences, get_batch


def test_shapes_and_padding():
    hist = {0: [1, 2, 3, 4], 1: [5, 6]}
    X, Y = build_training_sequences(hist, max_len=5, pad_id=0)
    assert X.shape == (2, 5) and Y.shape == (2, 5)
    # target is input shifted by one
    assert X[0].tolist() == [1, 2, 3, 0, 0]
    assert Y[0].tolist() == [2, 3, 4, 0, 0]
    assert X[1].tolist() == [5, 0, 0, 0, 0]
    assert Y[1].tolist() == [6, 0, 0, 0, 0]


def test_crops_to_maxlen():
    hist = {0: list(range(1, 20))}          # 19 items
    X, Y = build_training_sequences(hist, max_len=5, pad_id=0)
    assert X.shape == (1, 5)
    # last max_len+1 = 6 items -> inputs are items 14..18, targets 15..19
    assert X[0].tolist() == [14, 15, 16, 17, 18]
    assert Y[0].tolist() == [15, 16, 17, 18, 19]


def test_skips_short_users():
    hist = {0: [1], 1: [2, 3]}
    X, Y = build_training_sequences(hist, max_len=4, pad_id=0)
    assert X.shape[0] == 1


def test_get_batch_reproducible():
    hist = {i: [i + 1, i + 2, i + 3] for i in range(50)}
    X, Y = build_training_sequences(hist, max_len=4)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    xb1, yb1 = get_batch(X, Y, 8, g1)
    xb2, yb2 = get_batch(X, Y, 8, g2)
    assert torch.equal(xb1, xb2) and torch.equal(yb1, yb2)
    assert xb1.shape == (8, 4)
