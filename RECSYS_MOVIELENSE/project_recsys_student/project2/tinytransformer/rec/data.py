"""MovieLens data acquisition and preprocessing.  **PROVIDED -- do not edit.**

This module is given to you complete. It handles the plumbing (downloading,
parsing, id-remapping, chronological sorting, train/val/test splitting) so you
can spend your effort on the modeling. Read it once so you know what shapes the
rest of the project consumes, then use it as a black box.

The important outputs (see :class:`MovieLensData`):
  * ``n_items``          -- number of distinct movies (ids run 1..n_items)
  * ``PAD_ID == 0``      -- reserved padding id (never a real movie)
  * ``train_histories``  -- {user_id: [item_id, ...]} chronological, held-out removed
  * ``test``             -- {user_id: held_out_last_item_id}
  * ``item_title`` / ``item_primary_genre`` -- for readable output and evaluation

Vocabulary convention: a TinyTransformerLM for this task is built with
``vocab_size = data.vocab_size`` (== n_items + 1), so item id ``i`` indexes row
``i`` of the embedding table and row 0 stays the pad slot.
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from dataclasses import dataclass, field
from urllib.request import urlopen

PAD_ID = 0
ML_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@dataclass
class MovieLensData:
    n_items: int
    vocab_size: int
    item_title: dict
    item_genres: dict
    item_primary_genre: dict
    train_histories: dict            # user -> [item_id, ...]  (held-out removed)
    full_histories: dict             # user -> [item_id, ...]  (everything, for baselines)
    val: dict                        # user -> held-out item (second to last)
    test: dict                       # user -> held-out item (last)
    max_len: int = 50
    pad_id: int = PAD_ID
    genres: list = field(default_factory=list)

    def title(self, item_id):
        return self.item_title.get(item_id, f"<item {item_id}>")

    def find_item(self, query):
        """Return the item id whose title contains ``query`` (case-insensitive)."""
        q = query.lower()
        hits = [(i, t) for i, t in self.item_title.items() if q in t.lower()]
        if not hits:
            raise KeyError(f"no movie title contains {query!r}")
        hits.sort(key=lambda it: len(it[1]))
        return hits[0][0]


# --------------------------------------------------------------------------- #
# Download / cache
# --------------------------------------------------------------------------- #
def download_movielens(dest_dir="data", url=ML_SMALL_URL):
    """Download and extract ml-latest-small into ``dest_dir`` (cached).

    Returns the path to the extracted folder. Needs normal internet access;
    it will just work on your laptop. If you are offline or behind a proxy that
    blocks the download, use :func:`make_synthetic_movielens` instead.
    """
    os.makedirs(dest_dir, exist_ok=True)
    folder = os.path.join(dest_dir, "ml-latest-small")
    if os.path.exists(os.path.join(folder, "ratings.csv")):
        return folder
    print(f"Downloading MovieLens from {url} ...")
    with urlopen(url) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(dest_dir)
    print(f"Extracted to {folder}")
    return folder


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def build_dataset(folder, min_rating=4.0, min_user_len=5, min_item_count=5,
                  max_len=50):
    """Turn raw MovieLens CSVs into a :class:`MovieLensData`.

    Steps: keep ratings >= ``min_rating`` (treat as positive/implicit feedback);
    drop movies seen fewer than ``min_item_count`` times and users with fewer
    than ``min_user_len`` positives; remap movie ids to a contiguous 1..N range
    (0 reserved for PAD); sort each user's list by timestamp; hold out each
    user's last item as test and second-to-last as val (leave-one-out).
    """
    ratings = _read_csv(os.path.join(folder, "ratings.csv"))
    movies = _read_csv(os.path.join(folder, "movies.csv"))

    # keep positive interactions
    pos = [(r["userId"], r["movieId"], float(r["timestamp"]))
           for r in ratings if float(r["rating"]) >= min_rating]

    # count movie popularity, drop rare movies
    item_count = {}
    for _, m, _ in pos:
        item_count[m] = item_count.get(m, 0) + 1
    pos = [(u, m, t) for (u, m, t) in pos if item_count[m] >= min_item_count]

    # group by user, sort chronologically, drop short users
    by_user = {}
    for u, m, t in pos:
        by_user.setdefault(u, []).append((t, m))
    by_user = {u: sorted(v) for u, v in by_user.items() if len(v) >= min_user_len}

    # remap movie ids present in the surviving data -> 1..N
    present = sorted({m for v in by_user.values() for _, m in v})
    raw2idx = {m: i + 1 for i, m in enumerate(present)}   # 0 is PAD
    n_items = len(present)

    title_by_raw = {r["movieId"]: r["title"] for r in movies}
    genres_by_raw = {r["movieId"]: r["genres"].split("|") for r in movies}
    item_title, item_genres, item_primary = {}, {}, {}
    genre_set = set()
    for raw, idx in raw2idx.items():
        item_title[idx] = title_by_raw.get(raw, f"movie {raw}")
        g = genres_by_raw.get(raw, ["(no genres listed)"])
        item_genres[idx] = g
        item_primary[idx] = g[0] if g else "(no genres listed)"
        genre_set.update(g)

    # remap users to ints, build histories, leave-one-out split
    full_histories, train_histories, val, test = {}, {}, {}, {}
    for uid, (u, v) in enumerate(sorted(by_user.items())):
        seq = [raw2idx[m] for _, m in v]
        full_histories[uid] = seq
        test[uid] = seq[-1]
        val[uid] = seq[-2]
        train_histories[uid] = seq[:-2]        # history the model may learn from

    return MovieLensData(
        n_items=n_items, vocab_size=n_items + 1,
        item_title=item_title, item_genres=item_genres,
        item_primary_genre=item_primary,
        train_histories=train_histories, full_histories=full_histories,
        val=val, test=test, max_len=max_len, genres=sorted(genre_set),
    )


def load_movielens(dest_dir="data", **kwargs):
    """Convenience: download (if needed) and build the dataset in one call."""
    folder = download_movielens(dest_dir)
    return build_dataset(folder, **kwargs)


# --------------------------------------------------------------------------- #
# Offline synthetic dataset (same schema) -- for testing / no-internet use
# --------------------------------------------------------------------------- #
def make_synthetic_movielens(n_genres=6, items_per_genre=40, n_users=800,
                             hist_len=25, seed=0, max_len=50):
    """A tiny MovieLens-shaped dataset with planted *genre* and *sequential*
    structure.

    Each movie belongs to one genre, and within a genre the movies form a soft
    chain (think a franchise / release order): after one movie you tend to watch
    the next one in the chain. Users stick mostly to one genre, so:

      * item embeddings should cluster by genre (co-occurrence), and
      * a *sequential* model can beat a bag-of-co-occurrence baseline by using
        the chain order -- the transformer earns its keep.

    Same :class:`MovieLensData` schema as the real loader, so every downstream
    function works unchanged.
    """
    import random
    rng = random.Random(seed)

    n_items = n_genres * items_per_genre
    genres = [f"Genre{g}" for g in range(n_genres)]
    item_primary, item_genres, item_title = {}, {}, {}
    genre_items = {g: [] for g in range(n_genres)}
    for idx in range(1, n_items + 1):
        g = (idx - 1) // items_per_genre
        item_primary[idx] = genres[g]
        item_genres[idx] = [genres[g]]
        item_title[idx] = f"{genres[g]} Movie {idx}"
        genre_items[g].append(idx)          # chain order within the genre

    full_histories, train_histories, val, test = {}, {}, {}, {}
    for u in range(n_users):
        home = rng.randrange(n_genres)
        g = home
        pos = rng.randrange(items_per_genre)
        seq = []
        for _ in range(hist_len):
            seq.append(genre_items[g][pos])
            r = rng.random()
            if r < 0.70:                       # advance along the genre chain
                pos = (pos + 1) % items_per_genre
            elif r < 0.85:                     # random jump within the genre
                pos = rng.randrange(items_per_genre)
            else:                              # occasionally switch genre
                g = home if rng.random() < 0.5 else rng.randrange(n_genres)
                pos = rng.randrange(items_per_genre)
        full_histories[u] = seq
        test[u] = seq[-1]
        val[u] = seq[-2]
        train_histories[u] = seq[:-2]

    return MovieLensData(
        n_items=n_items, vocab_size=n_items + 1,
        item_title=item_title, item_genres=item_genres,
        item_primary_genre=item_primary,
        train_histories=train_histories, full_histories=full_histories,
        val=val, test=test, max_len=max_len, genres=genres,
    )
