'''
- load retrieval queries, corpus and qrels for a split
- encode queries and documents with each biencoder model
- rank all candidate docs by cosine similarity.
- evaluate on queries_with_relevant_docs only.
- save results on top 50 predictions.
'''
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import BIENCODER_MODELS, PROCESSED_DIR, PREDICTIONS_DIR, TABLES_DIR, ensure_directories
from src.utils import print_header, save_json

EVAL_SPLIT = "test"
TOP_K_VALUES = [5, 10, 20, 50]

def load_retrieval_artifacts(split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    queries_path = PROCESSED_DIR / f"retrieval_queries_{split_name}.csv"
    corpus_path = PROCESSED_DIR / f"retrieval_corpus_{split_name}.csv"
    qrels_path = PROCESSED_DIR / f"retrieval_qrels_{split_name}.csv"

    queries = pd.read_csv(queries_path)
    corpus = pd.read_csv(corpus_path)
    qrels = pd.read_csv(qrels_path)

    return queries, corpus, qrels

def validate_artifacts(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    qrels: pd.DataFrame,
) -> None:
    query_required = {"query_id", "query_text"}
    corpus_required = {"doc_id", "doc_text"}
    qrels_required = {"query_id", "doc_id", "relevance", "score"}

    missing_queries = query_required.difference(queries.columns)
    missing_corpus = corpus_required.difference(corpus.columns)
    missing_qrels = qrels_required.difference(qrels.columns)

    if missing_queries:
        raise ValueError(f"Missing query columns: {sorted(missing_queries)}")
    if missing_corpus:
        raise ValueError(f"Missing corpus columns: {sorted(missing_corpus)}")
    if missing_qrels:
        raise ValueError(f"Missing qrels columns: {sorted(missing_qrels)}")

def build_relevance_lookup(qrels: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Build mapping:
        query_id -> {doc_id: score}
    Only queries with at least one relevant doc appear here.
    """
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
    if rels.size == 0:
        return 0.0
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

def rank_documents(
    model: SentenceTransformer,
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode all queries and documents once and return:
    - query embeddings
    - document embeddings
    """
    query_embeddings = model.encode(
        queries["query_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    doc_embeddings = model.encode(
        corpus["doc_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return query_embeddings, doc_embeddings

def evaluate_single_model(
    model_name: str,
    model_path: str,
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    qrels: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    print_header(f"Evaluating Retrieval: {model_name}")
    model = SentenceTransformer(model_path)

    relevance_lookup = build_relevance_lookup(qrels)
    valid_query_ids = sorted(relevance_lookup.keys())

    filtered_queries = queries[queries["query_id"].isin(valid_query_ids)].reset_index(drop=True)
    if filtered_queries.empty:
        raise ValueError("No valid queries with relevant documents were found for evaluation.")

    query_embeddings, doc_embeddings = rank_documents(model, filtered_queries, corpus)

    similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)

    doc_ids = corpus["doc_id"].tolist()
    prediction_rows: list[dict[str, Any]] = []

    recall_scores = {k: [] for k in TOP_K_VALUES}
    ndcg_scores = {k: [] for k in TOP_K_VALUES}
    reciprocal_ranks: list[float] = []

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
                    "model_name": model_name,
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
        "model_name": model_name,
        "model_path": model_path,
        "eval_split": EVAL_SPLIT,
        "num_queries_total": int(len(queries)),
        "num_queries_evaluated": int(len(filtered_queries)),
        "num_docs": int(len(corpus)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }

    for k in TOP_K_VALUES:
        results[f"recall@{k}"] = float(np.mean(recall_scores[k]))
        results[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k]))

    predictions_df = pd.DataFrame(prediction_rows)
    return results, predictions_df

def main() -> None:
    ensure_directories()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print_header(f"Load Retrieval Artifacts ({EVAL_SPLIT})")
    queries, corpus, qrels = load_retrieval_artifacts(EVAL_SPLIT)
    validate_artifacts(queries, corpus, qrels)

    print(f"Queries shape: {queries.shape}")
    print(f"Corpus shape:  {corpus.shape}")
    print(f"Qrels shape:   {qrels.shape}")

    all_results: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []

    for model_name, model_path in BIENCODER_MODELS.items():
        result, predictions_df = evaluate_single_model(
            model_name=model_name,
            model_path=model_path,
            queries=queries,
            corpus=corpus,
            qrels=qrels,
        )
        all_results.append(result)
        all_predictions.append(predictions_df)

    results_df = pd.DataFrame(all_results).sort_values(
        by=["recall@50", "mrr", "ndcg@10"],
        ascending=False,
    ).reset_index(drop=True)

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results_path = TABLES_DIR / f"retrieval_results_{EVAL_SPLIT}.csv"
    predictions_path = PREDICTIONS_DIR / f"retrieval_top50_predictions_{EVAL_SPLIT}.csv"
    summary_path = TABLES_DIR / f"retrieval_results_{EVAL_SPLIT}.json"

    results_df.to_csv(results_path, index=False, encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")
    save_json(
        {
            "eval_split": EVAL_SPLIT,
            "results": results_df.to_dict(orient="records"),
            "best_model_by_recall_at_50": results_df.iloc[0]["model_name"] if not results_df.empty else None,
        },
        summary_path,
    )

    print_header("Retrieval Results")
    print(results_df)

    print(f"Saved retrieval results to: {results_path}")
    print(f"Saved top-50 predictions to: {predictions_path}")
    print(f"Saved retrieval summary to: {summary_path}")

if __name__ == "__main__":
    main()