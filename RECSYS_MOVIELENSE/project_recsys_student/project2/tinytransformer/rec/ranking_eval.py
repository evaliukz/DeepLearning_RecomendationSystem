"""Leave-one-out ranking evaluation harness.  PART F (continued).

Ties the pieces together: for each user, use their history (everything but the
last item) as context, ask a recommender for its top-k, and score where the
held-out last item landed. Average over users to compare approaches.
"""

"""
推荐系统评估（Leave-One-Out Ranking Evaluation）
目的评估不同推荐算法（Popularity、ItemKNN、Transformer）的效果。

采用 Leave-One-Out Evaluation
    用户历史：[A, B, C, D]
    输入给推荐器：[A, B, C]
    真正答案（Ground Truth）D

推荐器返回：Top-K 推荐列表

然后检查：
    D 是否被推荐出来？
    排名是否靠前？

build_eval_pairs()

将每个用户历史拆分成：(history, true_item)
例如： [A,B,C,D]
变成:
    history = [A,B,C]
    true_item = D
最后得到所有用户的测试样本。


evaluate_ranking()
对于每个测试样本：
1. 根据 history 生成 Top-K 推荐列表
2. 与真实商品 true_item 比较
3. 计算三个评价指标：
    Hit@K → 真正商品是否出现在 Top-K 中？
    NDCG@K → 真正商品排得越靠前，分数越高。
    MRR → 真正商品第一次出现的位置倒数（越靠前越好）。
最后：对所有用户取平均，得到整个推荐系统的性能。


返回结果
{
    "hit@k": 平均 Hit@K,
    "ndcg@k": 平均 NDCG@K,
    "mrr": 平均 MRR,
    "n": 测试用户数量
}

用每个用户最后一次交互作为正确答案，检查推荐系统是否能够根据之前的历史，
把真正的下一件商品推荐到 Top-K，并统计平均推荐效果。
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
    if not eval_pairs:
        return {"hit@k": 0.0, "ndcg@k": 0.0, "mrr": 0.0, "n": 0}

    hits = 0.0
    ndcgs = 0.0
    mrrs = 0.0

    for history, true_item in eval_pairs:
        ranked = recommender.recommend(history, k, exclude_seen=True)
        hits += hit_at_k(ranked, true_item, k)
        ndcgs += ndcg_at_k(ranked, true_item, k)
        mrrs += mrr(ranked, true_item)

    n = len(eval_pairs)
    return {"hit@k": hits / n, "ndcg@k": ndcgs / n, "mrr": mrrs / n, "n": n}

