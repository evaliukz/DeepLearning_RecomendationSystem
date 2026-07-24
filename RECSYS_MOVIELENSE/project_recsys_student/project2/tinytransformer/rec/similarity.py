"""The seed-program similarity tool.  PART D.

Given a trained model's item embeddings, find the movies most similar to a seed
movie by cosine similarity in embedding space. This is unsupervised retrieval:
no genre labels are used -- similarity is inferred purely from the geometry the
model learned.
"""

import torch


def similar_items(seed_id, emb, k=10):
    """Return the ``k`` items most cosine-similar to ``seed_id``.

    Args:
        seed_id: item id (1..n_items) of the seed movie.
        emb:     (n_items, d_model) embedding matrix from ``item_embeddings``
                 (row r == item id r+1).
        k:       number of neighbours to return.

    Returns:
        list of ``(item_id, score)`` pairs, highest score first, excluding the
        seed itself.
    """
    ### BEGIN CODE EDIT ###
    raise NotImplementedError("YOUR CODE HERE")
    ### END CODE EDIT ###


def recommend_similar(seed, data, emb, k=10):
    """**PROVIDED** convenience wrapper: seed by title (or id) -> titles.

    ``seed`` may be an item id or a substring of a title. Returns a list of
    ``(title, primary_genre, score)`` for nice printing in the notebook.
    """
    seed_id = seed if isinstance(seed, int) else data.find_item(seed)
    out = []
    for item_id, score in similar_items(seed_id, emb, k):
        out.append((data.title(item_id), data.item_primary_genre.get(item_id), score))
    return seed_id, out
