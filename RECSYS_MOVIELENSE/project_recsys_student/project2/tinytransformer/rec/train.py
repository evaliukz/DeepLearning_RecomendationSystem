"""Training loop for the sequential recommender.  PART B.

This is the Project-1 training loop with two differences: the "vocabulary" is
the item set, and padded target positions are ignored in the loss via
``ignore_index=pad_id``. The model is your ``TinyTransformerLM`` used unchanged.
"""

import torch
import torch.nn as nn

from .dataset import get_batch


def train_seqrec(model, X, Y, *, steps=800, batch_size=64, lr=3e-3, pad_id=0,
                 seed=1234, log_every=100, log=True):
    """Train ``model`` on padded (X, Y) next-item tensors.

    Given set-up: a seeded generator, Adam, and a CrossEntropyLoss that ignores
    ``pad_id`` targets. For each step:
        1. xb, yb = get_batch(X, Y, batch_size, g)
        2. zero grads
        3. logits = model(xb)                       -> (B, T, vocab)
        4. loss = CE(logits.reshape(-1, vocab), yb.reshape(-1))   (pad ignored)
        5. backward, step
        6. record loss.item()

    Returns ``{"losses": [...], "final_loss": float}``.
    """
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    model.train()
    losses = []
    ### BEGIN CODE EDIT ###
    raise NotImplementedError("YOUR CODE HERE")
    ### END CODE EDIT ###
    return {"losses": losses, "final_loss": losses[-1]}
