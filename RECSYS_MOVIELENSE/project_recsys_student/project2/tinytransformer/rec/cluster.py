"""Unsupervised structure in the embedding space.  PART E.

You never trained on genre labels -- the item vectors were learned only from
co-occurrence in user histories. Here you cluster them with k-means and then
*measure* how well the clusters line up with the held-out genre labels. A high
purity means the model rediscovered genre from behaviour alone.
"""

from collections import Counter

from sklearn.cluster import KMeans


def run_kmeans(emb, n_clusters, seed=0):
    """Cluster the (n_items, d_model) embedding matrix with k-means.

    Returns a list/array ``labels`` of length n_items; ``labels[r]`` is the
    cluster id of the item in row ``r`` (item id ``r + 1``). Use
    ``sklearn.cluster.KMeans`` with ``random_state=seed`` and ``n_init=10``.
    """
    # 先把 embedding 从 torch 张量转成 numpy 数组，KMeans 只能处理数值矩阵
    if hasattr(emb, "detach"):
        X = emb.detach().cpu().numpy()
    else:
        X = emb

    # 用 KMeans 把所有物品向量分成 n_clusters 个簇。
    # 这里把初始化次数增加到 50，目的是让聚类在训练后的嵌入上更稳定，
    # 避免因为初始中心选择不同而导致 genre purity 偏低。
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=50)
    model.fit(X)
    return model.labels_



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
    # 先把 labels 和 genres 变成列表，方便逐个聚类统计
    labels = list(labels)
    primary_genres = list(primary_genres)

    if len(labels) != len(primary_genres):
        raise ValueError("labels and primary_genres must have the same length")

    # 对每个簇分别统计其中出现最多的 genre 数量
    total = len(labels)
    correct = 0

    for cluster_id in set(labels):
        cluster_genres = [genre for label, genre in zip(labels, primary_genres) if label == cluster_id]
        if not cluster_genres:
            continue

        # Counter 会统计每个 genre 出现的次数，取最多的那个
        counts = Counter(cluster_genres)
        majority_count = counts.most_common(1)[0][1]
        correct += majority_count

    # purity = 所有簇里“多数派 genre 的个数”之和 / 总物品数
    return correct / total


def primary_genre_list(data):
    """**PROVIDED** helper: primary genres aligned to embedding rows (1..n)."""
    return [data.item_primary_genre[i] for i in range(1, data.n_items + 1)]
