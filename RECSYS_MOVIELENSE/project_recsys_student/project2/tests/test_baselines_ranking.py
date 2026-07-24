from tinytransformer.rec import (PopularityRecommender, ItemItemKNN,
                                 TransformerRecommender, build_eval_pairs,
                                 evaluate_ranking)


def test_popularity_fit_and_recommend():
    hist = {0: [1, 1, 2], 1: [1, 3], 2: [1, 2]}   # item 1 most popular
    rec = PopularityRecommender().fit(hist)
    assert rec.ranking[0] == 1
    out = rec.recommend([1], k=2, exclude_seen=True)
    assert 1 not in out and len(out) == 2


def test_itemitem_recommends_cooccurring():
    # items 1 and 2 always co-occur; 3 is separate
    hist = {u: [1, 2] for u in range(10)}
    hist.update({u: [3] for u in range(10, 15)})
    rec = ItemItemKNN().fit(hist)
    out = rec.recommend([1], k=1, exclude_seen=True)
    assert out[0] == 2


def test_evaluate_ranking_keys(synth):
    pairs = build_eval_pairs(synth)[:50]
    rec = PopularityRecommender().fit(synth.train_histories)
    res = evaluate_ranking(rec, pairs, k=10)
    assert set(res) == {"hit@k", "ndcg@k", "mrr", "n"}
    assert 0.0 <= res["hit@k"] <= 1.0


def test_transformer_beats_baselines(trained, synth):
    # synthetic data has genre clustering AND sequential (chain) structure, so
    # the ordering should be: transformer > item-item kNN > popularity.
    model, _ = trained
    pairs = build_eval_pairs(synth)[:150]
    r_pop = evaluate_ranking(PopularityRecommender().fit(synth.train_histories), pairs, k=10)
    r_knn = evaluate_ranking(ItemItemKNN().fit(synth.train_histories), pairs, k=10)
    r_tr = evaluate_ranking(TransformerRecommender(model, pad_id=synth.pad_id), pairs, k=10)
    assert r_knn["hit@k"] > r_pop["hit@k"]
    assert r_tr["hit@k"] >= r_knn["hit@k"]
