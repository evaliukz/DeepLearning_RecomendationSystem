"""Non-transformer baselines to compare against.  PART F.

Every recommender here exposes the same interface so the evaluation loop can
treat them uniformly:

    rec.fit(histories)                     # learn from {user: [item, ...]}
    rec.recommend(history, k, exclude_seen=True)  -> [item_id, ...] length k

``TransformerRecommender`` (PROVIDED) wraps your trained model in the same
interface. You implement the two classical baselines.
"""

"""
为了比较 Transformer 推荐系统是否真的更好，需要准备几个经典 Baseline。

这里实现了三个推荐器，它们都遵循相同接口：

    rec.fit(histories)
        ↓
    从所有用户历史中学习

    rec.recommend(history, k)
        ↓
    根据某个用户历史推荐 Top-K 商品

这样 Evaluation 时，不需要关心算法细节，只需要统一调用：

    rec.fit(...)
    rec.recommend(...)

即可公平比较不同算法。

Popularity：
    "大家都喜欢什么，我就推荐什么。"

Item-Item KNN：
    "和你看过商品相似的商品，我推荐给你。"

Transformer：
    "根据你的完整历史行为，
    我预测你下一步最可能喜欢什么。"

后续 Evaluation 会使用统一接口：

    fit()
    recommend()

来公平比较这三种推荐算法的效果。
"""

import torch


class PopularityRecommender:
    """Recommend the globally most-popular items. A hard-to-embarrass baseline."""

    def __init__(self):
        self.ranking = []          # item ids, most popular first

    def fit(self, histories):
        """Count how often each item appears across all histories and store a
        popularity-sorted list of item ids in ``self.ranking``."""
        ### BEGIN CODE EDIT ###
        # 统计每个物品在所有用户历史里出现的次数
        counts = {}
        for seq in histories.values():
            for item_id in seq:
                counts[item_id] = counts.get(item_id, 0) + 1

        # 按出现次数从高到低排序；如果次数相同，则按物品 ID 从小到大排
        self.ranking = [
            item_id for item_id, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return self
        ### END CODE EDIT ###

    def recommend(self, history, k=10, exclude_seen=True):
        """Return the top-k popular items, optionally skipping items in
        ``history``."""
        ### BEGIN CODE EDIT ###
        # 先把当前用户历史中的物品记成“已看过”集合
        seen = set(history) if history else set()

        # 从最流行的物品开始挑选，必要时跳过已看过的物品
        candidates = []
        for item_id in self.ranking:
            if exclude_seen and item_id in seen:
                continue
            candidates.append(item_id)
            if len(candidates) >= k:
                break

        return candidates
        ### END CODE EDIT ###


class ItemItemKNN:
    """Item-item collaborative filtering with cosine similarity.

    Build a binary user x item matrix, cosine-normalize the item columns, and
    score a candidate by how similar it is to the items already in the user's
    history. This is the classic 'people who watched these also watched...'.
    """

    def __init__(self, topk_history=20):
        self.topk_history = topk_history
        self.sim = None            # (n_items+1, n_items+1) dense cosine sims
        self.n_items = 0

    def fit(self, histories):
        """Compute item-item cosine similarities from co-occurrence.

        Build vectors ``a[item]`` over users (1 if the user has the item), then
        ``sim[i, j] = cos(a[i], a[j])``. Store in ``self.sim`` indexed by item
        id (row/col 0 is the unused pad slot)."""
        ### BEGIN CODE EDIT ###
        # 先收集所有出现过的物品 ID，并把它们按从小到大排序
        item_ids = sorted({item_id for seq in histories.values() for item_id in seq if item_id > 0})

        if not item_ids:
            self.sim = torch.zeros((1, 1), dtype=torch.float32)
            self.n_items = 0
            return self

        # 构造“用户 x 物品”的二值矩阵：用户是否交互过某个物品
        user_ids = sorted(histories.keys())
        vectors = {}
        for item_id in item_ids:
            vec = torch.zeros(len(user_ids), dtype=torch.float32)
            for idx, user_id in enumerate(user_ids):
                if item_id in set(histories[user_id]):
                    vec[idx] = 1.0
            vectors[item_id] = vec

        # 建立相似度矩阵，索引为物品 ID，0 留给 pad
        size = max(item_ids) + 1
        self.sim = torch.zeros((size, size), dtype=torch.float32)
        self.n_items = max(item_ids)

        for i in item_ids:
            for j in item_ids:
                vi = vectors[i]
                vj = vectors[j]
                norm_i = vi.norm()
                norm_j = vj.norm()
                if norm_i == 0 or norm_j == 0:
                    sim = 0.0
                else:
                    sim = (vi * vj).sum().item() / (norm_i * norm_j)
                self.sim[i, j] = sim

        return self
        ### END CODE EDIT ###

    def recommend(self, history, k=10, exclude_seen=True):
        """Score every item by summed similarity to the last ``topk_history``
        items of ``history``; return the top-k (excluding seen items)."""
        ### BEGIN CODE EDIT ###
        if self.sim is None:
            return []

        seen = set(history) if history else set()
        ctx = list(history)[-self.topk_history:] if history else []

        scores = {}
        for item_id in range(1, self.n_items + 1):
            if exclude_seen and item_id in seen:
                continue
            score = 0.0
            for ctx_item in ctx:
                if ctx_item <= 0 or ctx_item > self.n_items:
                    continue
                score += float(self.sim[ctx_item, item_id])
            scores[item_id] = score

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [item_id for item_id, _ in ranked[:k]]
        ### END CODE EDIT ###


class TransformerRecommender:
    """**PROVIDED** wrapper: rank items by your trained model's next-item logits."""

    def __init__(self, model, pad_id=0):
        self.model = model
        self.pad_id = pad_id

    def fit(self, histories):
        return self                     # already trained

    @torch.no_grad()
    def recommend(self, history, k=10, exclude_seen=True):
        self.model.eval()
        if not history:
            return []
        ctx = list(history)[-self.model.block_size:]
        x = torch.tensor([ctx], dtype=torch.long)
        logits = self.model(x)[0, -1]               # (vocab,)
        logits[self.pad_id] = float("-inf")
        if exclude_seen:
            for it in set(history):
                logits[it] = float("-inf")
        top = torch.topk(logits, k).indices.tolist()
        return [int(i) for i in top]
