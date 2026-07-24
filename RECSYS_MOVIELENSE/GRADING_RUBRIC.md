# Project 2 — Grading Rubric (for TAs)

**Total: 100 points.**  70 autograded (public + hidden pytest), 10 for a sane
end-to-end run, 20 for the written analysis. Everything runs on CPU in a few
minutes; there is no GPU or large-download requirement for grading (the
autograder uses the synthetic dataset, which is deterministic).

## How to grade

```bash
cd project2
pip install -r requirements.txt && pip install -e .
pytest tests -q                 # public suite (what students see)
pytest hidden_tests -q          # hidden suite (instructor-only; same components, more cases)
jupyter nbconvert --to notebook --execute notebooks/project2_movielens.ipynb  # for the run + analysis
```

The autograder imports **only** `tinytransformer.rec.*`; it does not depend on
the student's transformer core. Students may keep the shipped **reference**
`tinytransformer/` or drop in their own Project-1 code — **either is allowed and
neither affects the Project-2 score.** Do not deduct for Project-1 bugs here; if
a student's own core is broken, they were told to keep the reference files.

Determinism: the fixtures seed `torch.manual_seed(0)` and the synthetic data is
seeded, so autograded numbers are reproducible. Cosine/k-means results can vary
by a hair across BLAS builds — thresholds below already include margin, so do not
hand-tighten them.

---

## Autograded components — 70 pts

Award full marks if the mapped tests pass; use the partial-credit column when
some pass. "Hidden" cases probe the same function with different shapes/edge
cases.

### Part A — histories → tensors — 12 pts  (`rec/dataset.py`)
| Item | Pts | Pass criteria | Partial credit |
|------|----:|---------------|----------------|
| `build_training_sequences` | 8 | right-padding, one-step shift, crop to last `max_len+1`, skips users with <2 items | 4 if padding/shift correct but crop or skip wrong |
| `get_batch` | 4 | correct shape; reproducible under a seeded `Generator` | 2 if shape right but not reproducible |

### Part B — training loop — 12 pts  (`rec/train.py`)
| Item | Pts | Pass criteria | Partial credit |
|------|----:|---------------|----------------|
| `train_seqrec` | 12 | correct forward/loss/backward/step order; uses `ignore_index=pad_id`; loss falls markedly; returns `{"losses","final_loss"}` | 8 if it trains but forgets to ignore pad (loss still falls but purity/ranking degrade); 4 if it runs but loss does not fall |

### Part C — embeddings — 8 pts  (`rec/embeddings.py`)
| Item | Pts | Pass criteria | Partial credit |
|------|----:|---------------|----------------|
| `item_embeddings` | 8 | drops the pad row (shape `(n_items, d)`, row r ↔ item r+1); supports `which="output"/"input"`; `normalize` gives unit rows | 4 if correct table but pad row **not** dropped (off-by-one), 6 if `normalize` missing only |

### Part D — similarity tool — 10 pts  (`rec/similarity.py`)
| Item | Pts | Pass criteria | Partial credit |
|------|----:|---------------|----------------|
| `similar_items` | 10 | cosine (not raw dot) top-k; excludes the seed; sorted descending; returns `(item_id, score)` with correct id mapping | 6 if neighbours correct but seed not excluded or ids off-by-one; 4 if uses raw dot product |

### Part E — unsupervised structure — 14 pts  (`rec/cluster.py`)
| Item | Pts | Pass criteria | Partial credit |
|------|----:|---------------|----------------|
| `run_kmeans` | 6 | returns per-item labels of length `n_items` | 3 if runs but wrong length/alignment |
| `genre_purity` | 8 | majority-genre-per-cluster fraction; `=1.0` when clusters match genre, `=0.5` on the mixed toy case; learned output-embedding purity `> 0.7` | 4 if formula correct on toy cases but mis-aligned to rows |

### Part F — baselines & evaluation — 14 pts  (`rec/baselines.py`, `rec/ranking_eval.py`)
| Item | Pts | Pass criteria | Partial credit |
|------|----:|---------------|----------------|
| `PopularityRecommender` | 4 | popularity-sorted; excludes seen items | 2 if ranking right but doesn't exclude seen |
| `ItemItemKNN` | 6 | cosine item-item sim from co-occurrence; recommends co-occurring items; excludes seen | 3 if sim built but scoring/exclusion wrong |
| `evaluate_ranking` | 4 | correct hit@k/ndcg@k/MRR averaging; returns the four keys | 2 if metrics summed but not averaged |

---

## End-to-end run — 10 pts

From the executed notebook (or a fresh run):

- **4** — training loss falls substantially (e.g. by >1.5 on real data / to <0.5 on synthetic).
- **3** — k-means genre purity on the **output** embedding is high (≫ chance).
- **3** — comparison table shows **transformer ≥ item-item kNN > popularity** on hit@10.

Award proportionally if a student used real MovieLens and numbers differ
reasonably; the *ordering* and *direction* matter, not exact values.

---

## Written analysis — 20 pts  (notebook markdown)

### Q1 — input vs. output embedding — 8 pts
- **8** — states the output embedding recovers genre (and shows the purity gap), **and** explains *why*: the output row scores an item, so items predicted in the same contexts get similar output rows; the input table is only weakly shaped for this.
- **5** — correct which-is-better with the numbers but a thin/imprecise "why."
- **2** — reports numbers with no mechanism, or the reasoning is wrong.

### Q2 — when does the transformer earn its keep? — 8 pts
- **8** — explains the ordering via *order/recency* signal the transformer uses and the bag-of-items limitation of kNN, **and** names a real case where the simple baseline is hard to beat (cold-start / short histories, or popularity-dominated catalog).
- **5** — explains the ordering but the "hard-to-beat" case is missing or weak.
- **2** — restates the table without mechanism.

### Analysis quality & the content caveat — 4 pts
- **4** — engages with the "co-watching, not content" caveat (embeddings encode behaviour, so "similar" = "co-watched"); clear, correct writing.
- **2** — mentions it superficially.
- **0** — absent.

---

## Common failure modes (quick reference)

- **Pad row not dropped** in `item_embeddings` → every id is off by one; neighbours look random. (Part C / D.)
- **`ignore_index` omitted** in the loss → the model wastes capacity predicting PAD; ranking and purity sag even though loss "looks fine." (Part B.)
- **Raw dot product** instead of cosine in `similar_items` → popular (large-norm) items dominate every neighbour list. (Part D.)
- **Seen items not excluded** in ranking/baselines → inflated hit@k, and the held-out item can be crowded out. (Parts D/F.)
- **Using the input embedding** for the similarity tool → low purity. This is *not* a bug to penalize if the student **chose** it deliberately and discusses it in Q1; it is the point of the question.
- **Off-by-one shift** in `build_training_sequences` (input/target not aligned) → training silently learns identity-ish behaviour. (Part A.)

## Academic integrity

`torch.nn.MultiheadAttention`, `nn.Transformer*`, and
`F.scaled_dot_product_attention` remain disallowed for the model core (same as
Project 1). For the recommendation modules, `scikit-learn` is expected for
k-means and t-SNE; students should **not** import a ready-made recommender
library (e.g. `implicit`, `lightfm`, `surprise`) for the graded functions.
