"""t-SNE visualization of the item embeddings.  **PROVIDED.**

Plotting is not the learning objective, so this is given to you. You call it and
interpret the picture: do movies of the same genre land near each other, even
though genre was never a training signal?
"""


def tsne_embedding(emb, seed=0, perplexity=30):
    """Project (n_items, d_model) embeddings to 2-D with t-SNE. Returns (n,2)."""
    import numpy as np
    from sklearn.manifold import TSNE
    X = emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
    perplexity = min(perplexity, max(5, (len(X) - 1) // 3))
    ts = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    return ts.fit_transform(X)


def plot_tsne(emb, data, top_genres=8, seed=0, ax=None, save_path=None):
    """Scatter the t-SNE projection, coloured by each item's primary genre.

    Only the ``top_genres`` most common genres get their own colour; the rest
    are greyed out to keep the plot readable.
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    xy = tsne_embedding(emb, seed=seed)
    primary = [data.item_primary_genre[i] for i in range(1, data.n_items + 1)]
    common = [g for g, _ in Counter(primary).most_common(top_genres)]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    # grey background for the "other" genres
    other = [i for i, g in enumerate(primary) if g not in common]
    if other:
        ax.scatter(xy[other, 0], xy[other, 1], s=6, c="lightgrey", alpha=0.5,
                   label="other")
    for g in common:
        idx = [i for i, gg in enumerate(primary) if gg == g]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=10, alpha=0.8, label=g)
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_title("Item embeddings (t-SNE), coloured by primary genre")
    ax.set_xticks([]); ax.set_yticks([])
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
    return ax
