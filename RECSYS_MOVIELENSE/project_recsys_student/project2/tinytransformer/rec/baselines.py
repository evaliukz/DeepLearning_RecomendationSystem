"""Non-transformer baselines to compare against.  PART F.

Every recommender here exposes the same interface so the evaluation loop can
treat them uniformly:

    rec.fit(histories)                     # learn from {user: [item, ...]}
    rec.recommend(history, k, exclude_seen=True)  -> [item_id, ...] length k

``TransformerRecommender`` (PROVIDED) wraps your trained model in the same
interface. You implement the two classical baselines.
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
        raise NotImplementedError("YOUR CODE HERE")
        ### END CODE EDIT ###

    def recommend(self, history, k=10, exclude_seen=True):
        """Return the top-k popular items, optionally skipping items in
        ``history``."""
        ### BEGIN CODE EDIT ###
        raise NotImplementedError("YOUR CODE HERE")
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
        raise NotImplementedError("YOUR CODE HERE")
        ### END CODE EDIT ###

    def recommend(self, history, k=10, exclude_seen=True):
        """Score every item by summed similarity to the last ``topk_history``
        items of ``history``; return the top-k (excluding seen items)."""
        ### BEGIN CODE EDIT ###
        raise NotImplementedError("YOUR CODE HERE")
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
