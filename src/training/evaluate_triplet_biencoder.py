from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PREDICTIONS_DIR, PROCESSED_DIR, TABLES_DIR, ensure_directories
from src.utils import print_header, save_json

EVAL_SPLIT = "test"
TOP_K_VALUES = [5, 10, 20, 50]
MODEL_NAME = "gte_small_triplet"
MODEL_PATH = "models/triplet_finetuned/gte_small_triplet"

def load_retrieval_artifacts(split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    queries_path = PROCESSED_DIR / f"retrieval_queries_{split_name}.csv"
    corpus_path = PROCESSED_DIR / f"retrieval_corpus_{split_name}.csv"
    qrels_path = PROCESSED_DIR / f"retrieval_qrels_{split_name}.csv"

    queries = pd.read_csv(queries_path)
    corpus = pd.read_csv(corpus_path)
    qrels = pd.read_csv(qrels_path)

    return queries, corpus, qrels

def build_relevance_lookup(qrels: pd.DataFrame) -> dict[str, dict[str, float]]:
    relevance_lookup: dict[str, dict[str, float]] = {}
    for _, row in qrels.iterrows():
        qid = row["query_id"]
        did = row["doc_id"]
        score = float(row["score"])
        if qid not in relevance_lookup:
            relevance_lookup[qid] = {}
        relevance_lookup[qid][did] = score
    return relevance_lookup

def dcg_at_k(relevances: list[float], k: int) -> float:
    if k <= 0 or not relevances:
        return 0.0
    rels = np.asarray(relevances[:k], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))

def ndcg_at_k(ranked_doc_ids: list[str], relevant_docs: dict[str, float], k: int) -> float:
    actual_relevances = [relevant_docs.get(doc_id, 0.0) for doc_id in ranked_doc_ids[:k]]
    dcg = dcg_at_k(actual_relevances, k)
    ideal_relevances = sorted(relevant_docs.values(), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def recall_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    retrieved_top_k = set(ranked_doc_ids[:k])
    hits = len(retrieved_top_k.intersection(relevant_doc_ids))
    return hits / len(relevant_doc_ids)

def reciprocal_rank(ranked_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0

def main() -> None:
    ensure_directories()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print_header(f"Load Retrieval Artifacts ({EVAL_SPLIT})")
    queries, corpus, qrels = load_retrieval_artifacts(EVAL_SPLIT)

    relevance_lookup = build_relevance_lookup(qrels)
    valid_query_ids = sorted(relevance_lookup.keys())
    filtered_queries = queries[queries["query_id"].isin(valid_query_ids)].reset_index(drop=True)

    print_header("Load Fine-Tuned Model")
    model = SentenceTransformer(MODEL_PATH)

    print_header("Encode Queries and Corpus")
    query_embeddings = model.encode(
        filtered_queries["query_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    doc_embeddings = model.encode(
        corpus["doc_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)
    doc_ids = corpus["doc_id"].tolist()

    recall_scores = {k: [] for k in TOP_K_VALUES}
    ndcg_scores = {k: [] for k in TOP_K_VALUES}
    reciprocal_ranks: list[float] = []
    prediction_rows: list[dict[str, Any]] = []

    for query_idx, row in filtered_queries.iterrows():
        query_id = row["query_id"]
        query_text = row["query_text"]

        sims = similarity_matrix[query_idx]
        ranked_indices = np.argsort(-sims)
        ranked_doc_ids = [doc_ids[idx] for idx in ranked_indices]

        relevant_docs = relevance_lookup[query_id]
        relevant_doc_ids = set(relevant_docs.keys())

        for k in TOP_K_VALUES:
            recall_scores[k].append(recall_at_k(ranked_doc_ids, relevant_doc_ids, k))
            ndcg_scores[k].append(ndcg_at_k(ranked_doc_ids, relevant_docs, k))

        reciprocal_ranks.append(reciprocal_rank(ranked_doc_ids, relevant_doc_ids))

        for rank_position, doc_idx in enumerate(ranked_indices[:50], start=1):
            prediction_rows.append(
                {
                    "model_name": MODEL_NAME,
                    "query_id": query_id,
                    "query_text": query_text,
                    "doc_id": doc_ids[doc_idx],
                    "doc_text": corpus.iloc[doc_idx]["doc_text"],
                    "rank": rank_position,
                    "similarity": float(sims[doc_idx]),
                    "is_relevant": int(doc_ids[doc_idx] in relevant_doc_ids),
                    "graded_relevance": float(relevant_docs.get(doc_ids[doc_idx], 0.0)),
                }
            )
    results = {
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "eval_split": EVAL_SPLIT,
        "num_queries_total": int(len(queries)),
        "num_queries_evaluated": int(len(filtered_queries)),
        "num_docs": int(len(corpus)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }
    for k in TOP_K_VALUES:
        results[f"recall@{k}"] = float(np.mean(recall_scores[k]))
        results[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k]))

    results_df = pd.DataFrame([results])
    predictions_df = pd.DataFrame(prediction_rows)

    results_path = TABLES_DIR / f"triplet_retrieval_results_{EVAL_SPLIT}.csv"
    predictions_path = PREDICTIONS_DIR / f"triplet_top50_predictions_{EVAL_SPLIT}.csv"
    summary_path = TABLES_DIR / f"triplet_retrieval_results_{EVAL_SPLIT}.json"

    results_df.to_csv(results_path, index=False, encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")
    save_json(results, summary_path)

    print_header("Triplet Fine-Tuned Retrieval Results")
    print(results_df)
    print(f"Saved results to: {results_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()