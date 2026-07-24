import torch
from tinytransformer.rec import item_embeddings, similar_items, recommend_similar


def test_item_embeddings_shape(trained, synth):
    model, emb = trained
    assert emb.shape == (synth.n_items, model.head.weight.size(1))
    # default is the OUTPUT embedding; pad row dropped -> row 0 == item id 1
    assert torch.allclose(emb[0], model.head.weight.detach().cpu()[1])


def test_normalize_unit_norm(trained):
    model, _ = trained
    emb = item_embeddings(model, normalize=True)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_similar_items_excludes_seed_and_len(trained):
    _, emb = trained
    out = similar_items(3, emb, k=5)
    assert len(out) == 5
    ids = [i for i, _ in out]
    assert 3 not in ids
    # scores sorted descending
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_neighbours_share_genre(trained, synth):
    # planted structure: neighbours should mostly share the seed's genre.
    # average over several seeds to avoid single-seed noise.
    _, emb = trained
    total_same = 0
    seeds = [3, 25, 45, 65, 85]
    for seed in seeds:
        g = synth.item_primary_genre[seed]
        out = similar_items(seed, emb, k=10)
        total_same += sum(1 for i, _ in out if synth.item_primary_genre[i] == g)
    # chance would be ~2/10; learned output embeddings should be well above that
    assert total_same / len(seeds) >= 4.5
