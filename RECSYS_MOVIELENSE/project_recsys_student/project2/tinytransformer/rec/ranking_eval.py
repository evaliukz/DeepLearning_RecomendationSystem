"""Leave-one-out ranking evaluation harness.  PART F (continued).

Ties the pieces together: for each user, use their history (everything but the
last item) as context, ask a recommender for its top-k, and score where the
held-out last item landed. Average over users to compare approaches.
"""

from .metrics import hit_at_k, ndcg_at_k, mrr


def build_eval_pairs(data):
    """**PROVIDED**: list of (history, held_out_item) from the dataset.

    History is the full interaction list with the final (test) item removed;
    the removed item is the target to be recovered.
    """
    pairs = []
    for user, seq in data.full_histories.items():
        if len(seq) < 2:
            continue
        pairs.append((seq[:-1], seq[-1]))
    return pairs


def evaluate_ranking(recommender, eval_pairs, k=10):
    """Average hit@k, ndcg@k and MRR of ``recommender`` over ``eval_pairs``.

    For each (history, true_item):
        ranked = recommender.recommend(history, k, exclude_seen=True)
        accumulate hit_at_k(ranked, true_item, k),
                   ndcg_at_k(ranked, true_item, k),
                   mrr(ranked, true_item)
    Return a dict: ``{"hit@k": ..., "ndcg@k": ..., "mrr": ..., "n": N}``.
    """
    ### BEGIN CODE EDIT ###
    raise NotImplementedError("YOUR CODE HERE")
    ### END CODE EDIT ###
