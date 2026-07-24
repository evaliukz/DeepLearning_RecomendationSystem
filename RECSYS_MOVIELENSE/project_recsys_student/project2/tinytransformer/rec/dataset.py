"""Turn user histories into padded next-item training tensors.  PART A.

Key idea: a user's history ``[i1, i2, ..., iL]`` is treated exactly like a
sentence. The model predicts the next item at every position, so the training
input is ``[i1..i_{L-1}]`` and the target is ``[i2..iL]`` (shifted by one). We
right-pad every sequence to ``max_len`` with ``pad_id``.

Why right-padding needs no attention mask: the model is causal, so a real
position only ever attends to *earlier* positions -- and with padding on the
right, everything earlier is real. Pad positions are ignored in the loss
(``ignore_index=pad_id``), so they never affect training.
"""

import torch


def build_training_sequences(histories, max_len, pad_id=0):
    """Build right-padded (inputs, targets) tensors from user histories.

    Args:
        histories: {user: [item_id, ...]} chronological, ints in 1..n_items.
        max_len:   sequences are cropped to their last ``max_len + 1`` items
                   (so inputs/targets are at most ``max_len`` long) then padded.
        pad_id:    id used for padding (default 0).

    Returns:
        X: (N, max_len) LongTensor of inputs   (right-padded with pad_id)
        Y: (N, max_len) LongTensor of targets  (right-padded with pad_id)

    One row per user with at least 2 items. Users with <2 items are skipped.
    """
    ### BEGIN CODE EDIT ###
    raise NotImplementedError("YOUR CODE HERE")
    ### END CODE EDIT ###


def get_batch(X, Y, batch_size, generator=None):
    """Sample ``batch_size`` random rows from (X, Y).

    Returns (xb, yb), each (batch_size, max_len). Sampling with replacement is
    fine. Use ``generator`` for reproducibility.
    """
    ### BEGIN CODE EDIT ###
    raise NotImplementedError("YOUR CODE HERE")
    ### END CODE EDIT ###
