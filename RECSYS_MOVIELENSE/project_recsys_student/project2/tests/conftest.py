import pytest
import torch

from tinytransformer.rec import data as datamod


@pytest.fixture(scope="session")
def synth():
    """Small synthetic MovieLens-shaped dataset with planted genre structure."""
    return datamod.make_synthetic_movielens(
        n_genres=5, items_per_genre=20, n_users=200, hist_len=20, seed=0, max_len=30
    )


@pytest.fixture(scope="session")
def trained(synth):
    """A briefly-trained model + its item embeddings (session-scoped: trained once)."""
    from tinytransformer import TinyTransformerLM
    from tinytransformer.rec import build_training_sequences, train_seqrec, item_embeddings

    X, Y = build_training_sequences(synth.train_histories, synth.max_len, synth.pad_id)
    torch.manual_seed(0)                      # deterministic model init
    model = TinyTransformerLM(vocab_size=synth.vocab_size, d_model=32, n_heads=4,
                              n_layers=2, block_size=synth.max_len)
    train_seqrec(model, X, Y, steps=300, batch_size=64, lr=3e-3,
                 pad_id=synth.pad_id, log=False)
    emb = item_embeddings(model, normalize=False)
    return model, emb
