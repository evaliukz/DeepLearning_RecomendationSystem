"""Extract learned item embeddings from a trained model.  PART C.

Your model actually holds **two** vectors per item:

  * the *input* embedding  ``model.tok_emb.weight``  -- item as context
  * the *output* embedding ``model.head.weight``     -- item as prediction target
    (row ``i`` of the final linear layer is the vector whose dot product with a
    hidden state produces item ``i``'s logit)

Both are learned only from co-occurrence in user histories -- genre is never a
training signal. But they do not behave the same for similarity: items that are
*predicted in the same situations* end up with similar OUTPUT rows, so the
output embedding is what clusters cleanly by genre. Comparing the two is one of
the analysis questions -- that is why ``which`` is a parameter.
"""


def item_embeddings(model, which="output", normalize=False):
    """Return an (n_items, d_model) item-embedding matrix, dropping pad row 0.

    Args:
        model: a trained TinyTransformerLM (vocab_size == n_items + 1).
        which: ``"output"`` -> ``model.head.weight`` (default, best for
               similarity), or ``"input"`` -> ``model.tok_emb.weight``.
        normalize: if True, L2-normalize each row.

    Returns:
        emb: (n_items, d_model). Row ``r`` corresponds to item id ``r + 1``.
        Detached, on CPU.
    """
    ### BEGIN CODE EDIT ###
    raise NotImplementedError("YOUR CODE HERE")
    ### END CODE EDIT ###


def emb_row(item_id):
    """Map an item id (1..n_items) to its row index in ``item_embeddings``."""
    return item_id - 1
