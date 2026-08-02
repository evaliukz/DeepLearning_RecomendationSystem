"""The seed-program similarity tool.  PART D.

Given a trained model's item embeddings, find the movies most similar to a seed
movie by cosine similarity in embedding space. This is unsupervised retrieval:
no genre labels are used -- similarity is inferred purely from the geometry the
model learned.
"""

"""
根据训练好的 Item Embedding 寻找最相似的电影（Item Retrieval）

目的: Transformer 训练结束后，每个电影（或商品）都会学到一个 Embedding 向量。

例如:
    Toy Story      → [0.31, -0.22, ..., 0.15]
    Finding Nemo   → [0.29, -0.18, ..., 0.12]
    Star Wars      → [-0.54, 0.91, ..., -0.37]

这些向量不是人工设计的，也没有告诉模型电影属于什么 Genre。
模型只是根据用户观看历史（Co-occurrence）自动学习哪些电影经常出现在相似的用户行为中。
因此向量越接近,模型认为两个电影越相似

为什么可以推荐相似电影？

例如很多用户都看：Toy Story ->Finding Nemo
或者Toy Story->Monsters Inc
模型就会把这些电影学习成Embedding 很接近,因此给定 Toy Story就可以寻找Embedding 最接近 Toy Story 的电影作为推荐结果。

整个过程：
    Seed Movie
          ↓
    Item Embedding
          ↓
    Cosine Similarity
          ↓
    Top-K Most Similar Movies

为什么使用 Cosine Similarity?这里不用欧氏距离,而是使用Cosine Similarity余弦相似度
如果 Embedding 已经做了 L2 Normalize，那么Cosine Similarity向量点积
优点只比较方向是否一致，不受向量长度影响。

整个函数流程
Step1输入一个电影 id :seed_id = 25
Step2找到它对应的 Embedding: seed_vec = emb[24]因为item_id = row + 1,所以row = seed_id - 1
Step3计算这个电影和所有电影的 Cosine Similarity
    scores =cosine_similarity(所有电影,seed电影)分数越高，表示两个电影越相似。

Step4不能推荐自己。scores[row_idx] = -inf 这样 TopK 时自己一定不会被选中。

Step5寻找分数最高的 K 个电影：topk = torch.topk(scores, k)

Step6返回[(movie_id, similarity_score),...]
之后可以再根据 movie_id 查询电影标题和类型，变成最终推荐结果。


Transformer 训练完成后，每个电影都会学到一个 Embedding。
similar_items() 的作用就是：
    选择一个种子电影
            ↓
    计算它和所有电影的 Cosine Similarity
            ↓
    找到最相似的 Top-K 电影
            ↓
    作为推荐结果。

整个推荐过程完全基于模型学习到的 Embedding 几何结构（Geometry），
没有使用任何 Genre 标签，因此属于一种无监督（Unsupervised）的相似商品检索。
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
    # 把 item_id 转成 embedding 矩阵里的行号
    row_idx = seed_id - 1
    seed_vec = emb[row_idx]

    # 用余弦相似度衡量“哪个物品和种子物品最像”
    scores = torch.nn.functional.cosine_similarity(emb, seed_vec, dim=1)

    # 去掉自己本身
    scores[row_idx] = float("-inf")

    topk = torch.topk(scores, k=min(k, len(scores) - 1))
    return [(int(idx + 1), float(score)) for idx, score in zip(topk.indices.tolist(), topk.values.tolist())]


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
