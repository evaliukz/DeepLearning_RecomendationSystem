from tinytransformer.rec import run_kmeans, genre_purity, primary_genre_list


def test_kmeans_labels_length(trained, synth):
    _, emb = trained
    labels = run_kmeans(emb, n_clusters=len(synth.genres), seed=0)
    assert len(labels) == synth.n_items


def test_purity_perfect_when_clusters_match_genre():
    # 2 clusters, 2 genres, perfectly separated
    labels = [0, 0, 1, 1]
    primary = ["A", "A", "B", "B"]
    assert genre_purity(labels, primary) == 1.0


def test_purity_half_when_mixed():
    labels = [0, 0, 0, 0]
    primary = ["A", "A", "B", "B"]
    assert genre_purity(labels, primary) == 0.5


def test_learned_embeddings_recover_genre(trained, synth):
    _, emb = trained
    labels = run_kmeans(emb, n_clusters=len(synth.genres), seed=0)
    purity = genre_purity(labels, primary_genre_list(synth))
    assert purity > 0.7                       # much better than chance (~0.2)
