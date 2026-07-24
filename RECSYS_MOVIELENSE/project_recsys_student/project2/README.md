# Project 2 — Your Transformer, Now a Recommender

In Project 1 you built a GPT-style transformer and trained it to predict the
next **character**. In this project you point that *same* architecture at a
completely different problem: **sequential recommendation**. Treat each movie as
a "token" and each user's chronologically ordered watch history as a "sentence,"
and next-token prediction becomes **next-movie prediction**. You train on
MovieLens, pull the learned item embeddings out of the model, and build a tool
that — given a seed movie — returns similar movies. Then you study that
embedding space with the **unsupervised** methods from lecture and measure your
recommender against classic baselines.

The big idea: *recommendation is language modeling with a different vocabulary.*
You will feel your Project-1 code generalize with almost no changes.

## Use your own transformer (with a safety net)

This package ships a **reference** `tinytransformer/` (the Project-1 core, fully
implemented) so everything runs out of the box. But Project 2 is meant to run on
**your** Project-1 transformer:

> Replace the files in `tinytransformer/` (`attention.py`, `masking.py`,
> `positional.py`, `feedforward.py`, `block.py`, `model.py`) with your own
> Project-1 versions. If your Project-1 implementation had issues, **keep the
> reference files** — you will not lose any Project-2 credit for a Project-1 bug.
> The graded code for this project is entirely in `tinytransformer/rec/`.

Nothing in the model needs to change for this project. Because the model is
causal and we **right-pad** histories, a real position only ever attends to
earlier (real) positions, and padded targets are ignored in the loss — so no new
attention masking is required. Your Project-1 `forward` works unchanged.

## Setup

```bash
cd project2
python3 -m pip install -r requirements.txt
python3 -m pip install -e .       # makes `import tinytransformer` work everywhere
python3 -m pytest tests           # public tests (most fail until you implement things)
```

On macOS the bare `pip` / `pytest` commands are often not on your PATH — use
`python3 -m pip` and `python3 -m pytest` as shown above. If `pip install -e .`
complains about editable mode, upgrade pip once with
`python3 -m pip install --upgrade pip` (this project also ships a `setup.py` so
older pip works too).

Everything runs on CPU in a few minutes. No GPU required.

## What you implement

Search `tinytransformer/rec/` for `### BEGIN CODE EDIT ###` markers. The data
download/preprocessing (`data.py`), the ranking metrics (`metrics.py`), and the
t-SNE plotting (`viz.py`) are **given** — spend your effort on the modeling.

### Part A — histories → training tensors (`rec/dataset.py`)
| Function | What you implement |
|----------|--------------------|
| `build_training_sequences` | right-pad each user's history into (input, target) next-item tensors |
| `get_batch` | sample a reproducible minibatch |

### Part B — train the recommender (`rec/train.py`)
| Function | What you implement |
|----------|--------------------|
| `train_seqrec` | the training loop; next-item cross-entropy with padded targets ignored (`ignore_index=pad_id`) |

### Part C — pull out the embeddings (`rec/embeddings.py`)
| Function | What you implement |
|----------|--------------------|
| `item_embeddings` | return the per-item vectors; choose the **input** (`tok_emb`) or **output** (`head`) embedding |

### Part D — the seed-program similarity tool (`rec/similarity.py`)
| Function | What you implement |
|----------|--------------------|
| `similar_items` | top-k cosine neighbours of a seed movie in embedding space |

### Part E — unsupervised structure (`rec/cluster.py`)
| Function | What you implement |
|----------|--------------------|
| `run_kmeans` | k-means over the item embeddings |
| `genre_purity` | how well clusters line up with held-out genre labels |

### Part F — baselines & comparison (`rec/baselines.py`, `rec/ranking_eval.py`)
| Function | What you implement |
|----------|--------------------|
| `PopularityRecommender` | most-popular ranking |
| `ItemItemKNN` | item-item cosine collaborative filtering |
| `evaluate_ranking` | leave-one-out hit@k / ndcg@k / MRR over all users |

`TransformerRecommender` (your trained model wrapped in the same interface) is
provided so the comparison is apples-to-apples.

## The notebook

`notebooks/project2_movielens.ipynb` is the driver. It walks the whole pipeline —
get data, train, embeddings, t-SNE, the similarity demo, and the baseline
comparison table — and contains the **analysis questions** you must answer in
markdown. Two questions matter most:

1. **Input vs. output embedding.** Build `item_embeddings(model, which="input")`
   and `which="output")`, cluster each, and report `genre_purity`. Which one
   recovers genre, and *why*? (Think about what shapes each table during
   training.)
2. **When does the transformer earn its keep?** Your comparison table has
   popularity, item-item kNN, and the transformer. Explain the ordering you see,
   and name one situation where the simplest baseline would be hard to beat.

A caveat worth discussing in your answers: MovieLens has no plot text, so these
embeddings encode *co-watching behaviour*, not content. "Similar to Inception"
means "watched by the same people," which usually — but not always — lines up
with content.

## Data

`rec/data.load_movielens()` downloads **MovieLens ml-latest-small** (≈100k
ratings, released by GroupLens; ratings run through 2023) and preprocesses it.
Develop on this. If you are offline or behind a proxy, `make_synthetic_movielens()`
returns a dataset with the same schema and planted genre + sequential structure,
so every function still runs. You may optionally scale to a subsample of
MovieLens-32M — cap history length and subsample users to keep it CPU-friendly.

## Tests

`pytest tests` runs the **public** suite. Passing it is a good sign but not the
whole grade — a hidden suite checks the same components more thoroughly, and the
notebook's analysis answers are graded by hand. Write correct, general code.

```python
# quick end-to-end sanity check once everything is implemented
from tinytransformer import TinyTransformerLM
from tinytransformer.rec import (data, build_training_sequences, train_seqrec,
    item_embeddings, recommend_similar)

d = data.load_movielens()                      # or data.make_synthetic_movielens()
X, Y = build_training_sequences(d.train_histories, d.max_len, d.pad_id)
m = TinyTransformerLM(vocab_size=d.vocab_size, d_model=64, block_size=d.max_len)
train_seqrec(m, X, Y, steps=800)               # loss should fall a lot
emb = item_embeddings(m, which="output")
seed_id, hits = recommend_similar("Toy Story", d, emb, k=10)
for title, genre, score in hits:
    print(f"{score:.3f}  {title}  [{genre}]")
```
