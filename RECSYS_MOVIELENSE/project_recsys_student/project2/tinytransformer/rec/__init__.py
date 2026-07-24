"""tinytransformer.rec -- turn the Project-1 transformer into a movie recommender.

Pipeline:  data -> sequences -> train -> embeddings -> {similarity, clustering,
ranking vs baselines}.
"""

from . import data
from .dataset import build_training_sequences, get_batch
from .train import train_seqrec
from .embeddings import item_embeddings, emb_row
from .similarity import similar_items, recommend_similar
from .cluster import run_kmeans, genre_purity, primary_genre_list
from .baselines import PopularityRecommender, ItemItemKNN, TransformerRecommender
from .ranking_eval import build_eval_pairs, evaluate_ranking
from . import metrics
from . import viz

__all__ = [
    "data",
    "build_training_sequences", "get_batch",
    "train_seqrec",
    "item_embeddings", "emb_row",
    "similar_items", "recommend_similar",
    "run_kmeans", "genre_purity", "primary_genre_list",
    "PopularityRecommender", "ItemItemKNN", "TransformerRecommender",
    "build_eval_pairs", "evaluate_ranking",
    "metrics", "viz",
]
