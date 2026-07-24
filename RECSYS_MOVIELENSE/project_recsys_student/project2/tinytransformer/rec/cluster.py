"""Unsupervised structure in the embedding space.  PART E.

You never trained on genre labels -- the item vectors were learned only from
co-occurrence in user histories. Here you cluster them with k-means and then
*measure* how well the clusters line up with the held-out genre labels. A high
purity means the model rediscovered genre from behaviour alone.
"""


def run_kmeans(emb, n_clusters, seed=0):
    """Cluster the (n_items, d_model) embedding matrix with k-means.

    Returns a list/array ``labels`` of length n_items; ``labels[r]`` is the
    cluster id of the item in row ``r`` (item id ``r + 1``). Use
    ``sklearn.cluster.KMeans`` with ``random_state=seed`` and ``n_init=10``.
    """
    ### BEGIN CODE EDIT ###
    raise NotImplementedError("YOUR CODE HERE")
    ### END CODE EDIT ###


def genre_purity(labels, primary_genres):
    """Fraction of items whose cluster's majority genre matches their own.

    Args:
        labels: cluster id per row (row r == item id r+1), length n_items.
        primary_genres: list of primary-genre strings aligned to the SAME rows,
            i.e. ``primary_genres[r]`` is the primary genre of item ``r + 1``.

    Returns:
        purity in [0, 1]: sum over clusters of (size of the most common genre in
        that cluster), divided by the total number of items.
    """
    ### BEGIN CODE EDIT ###
    raise NotImplementedError("YOUR CODE HERE")
    ### END CODE EDIT ###


def primary_genre_list(data):
    """**PROVIDED** helper: primary genres aligned to embedding rows (1..n)."""
    return [data.item_primary_genre[i] for i in range(1, data.n_items + 1)]
