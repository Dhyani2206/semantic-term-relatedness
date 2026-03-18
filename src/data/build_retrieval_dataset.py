'''
Here we define:
query = term1
candidate corpus = terms available for retrieval
relevance = term2 with score above threshold (score >= 0.70)
qrels = for train/test/val

Fore each split build qrels from that split and build retrieval corpus from that split only 
to avoid contamination, respects grouped logic and clean evaluation.
'''
from __future__ import annotations
from typing import Any
import pandas as pd

from src.config import (
    PROCESSED_DIR,
    RETRIEVAL_POSITIVE_THRESHOLD,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    ensure_directories,
)
from src.utils import print_header, save_json

def validate_split_columns(df: pd.DataFrame, split_name: str) -> None:
    required = {"topic", "term1", "term2", "score", "relatedness_level"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {split_name} split: {sorted(missing)}"
        )

def build_query_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build unique query table from term1 values.
    """
    queries = (
        pd.DataFrame({"query_text": sorted(df["term1"].unique())})
        .reset_index(drop=True)
    )
    queries["query_id"] = [f"q_{i:06d}" for i in range(len(queries))]
    return queries[["query_id", "query_text"]]

def build_corpus_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build unique retrieval corpus from term2 values.
    """
    corpus = (
        pd.DataFrame({"doc_text": sorted(df["term2"].unique())})
        .reset_index(drop=True)
    )
    corpus["doc_id"] = [f"d_{i:06d}" for i in range(len(corpus))]
    return corpus[["doc_id", "doc_text"]]

def build_qrels(
    df: pd.DataFrame,
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    positive_threshold: float,
) -> pd.DataFrame:
    """
    Build retrieval qrels from semantic pairs.
    A pair is considered relevant if score >= positive_threshold.
    """
    positives = df[df["score"] >= positive_threshold].copy()
    qrels = (
        positives.merge(
            queries,
            left_on="term1",
            right_on="query_text",
            how="inner",
        )
        .merge(
            corpus,
            left_on="term2",
            right_on="doc_text",
            how="inner",
        )
    )

    qrels = qrels[
        ["query_id", "doc_id", "score", "topic", "term1", "term2", "relatedness_level"]
    ].copy()

    # If duplicate query-doc pairs somehow remain, keep the max score
    qrels = (
        qrels.groupby(
            ["query_id", "doc_id", "topic", "term1", "term2", "relatedness_level"],
            as_index=False
        )
        .agg(score=("score", "max"))
        .reset_index(drop=True)
    )

    # Binary relevance for retrieval evaluation
    qrels["relevance"] =  qrels["score"]

    return qrels[
        ["query_id", "doc_id", "relevance", "score", "topic", "term1", "term2", "relatedness_level"]
    ]

def summarize_retrieval_artifacts(
    split_name: str,
    df: pd.DataFrame,
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    qrels: pd.DataFrame,
    positive_threshold: float,
) -> dict[str, Any]:
    queries_with_relevant_docs = int(qrels["query_id"].nunique()) if not qrels.empty else 0
    total_queries = int(len(queries))
    queries_without_relevant_docs = total_queries - queries_with_relevant_docs

    avg_relevant_docs = (
        float(qrels.groupby("query_id")["doc_id"].nunique().mean())
        if not qrels.empty
        else 0.0
    )

    return {
        "split_name": split_name,
        "input_rows": int(len(df)),
        "unique_topics": int(df["topic"].nunique()),
        "unique_queries": total_queries,
        "unique_docs": int(len(corpus)),
        "positive_threshold": float(positive_threshold),
        "positive_pairs": int(len(qrels)),
        "queries_with_relevant_docs": queries_with_relevant_docs,
        "queries_without_relevant_docs": int(queries_without_relevant_docs),
        "avg_relevant_docs_per_query": avg_relevant_docs,
    }

def save_split_artifacts(
    split_name: str,
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    qrels: pd.DataFrame,
) -> None:
    queries_path = PROCESSED_DIR / f"retrieval_queries_{split_name}.csv"
    corpus_path = PROCESSED_DIR / f"retrieval_corpus_{split_name}.csv"
    qrels_path = PROCESSED_DIR / f"retrieval_qrels_{split_name}.csv"

    queries.to_csv(queries_path, index=False, encoding="utf-8")
    corpus.to_csv(corpus_path, index=False, encoding="utf-8")
    qrels.to_csv(qrels_path, index=False, encoding="utf-8")

    print(f"Saved queries to: {queries_path}")
    print(f"Saved corpus to:  {corpus_path}")
    print(f"Saved qrels to:   {qrels_path}")

def process_split(split_name: str, split_path: str | Any) -> dict[str, Any]:
    print_header(f"Processing Split: {split_name}")
    df = pd.read_csv(split_path)
    validate_split_columns(df, split_name)

    queries = build_query_table(df)
    corpus = build_corpus_table(df)
    qrels = build_qrels(
        df=df,
        queries=queries,
        corpus=corpus,
        positive_threshold=RETRIEVAL_POSITIVE_THRESHOLD,
    )
    save_split_artifacts(split_name, queries, corpus, qrels)

    summary = summarize_retrieval_artifacts(
        split_name=split_name,
        df=df,
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        positive_threshold=RETRIEVAL_POSITIVE_THRESHOLD,
    )

    print(
        f"{split_name.upper()} | "
        f"queries={summary['unique_queries']}, "
        f"docs={summary['unique_docs']}, "
        f"positive_pairs={summary['positive_pairs']}, "
        f"queries_with_relevant_docs={summary['queries_with_relevant_docs']}"
    )
    return summary

def main() -> None:
    ensure_directories()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    summaries = {
        "train": process_split("train", TRAIN_CSV_PATH),
        "val": process_split("val", VAL_CSV_PATH),
        "test": process_split("test", TEST_CSV_PATH),
    }
    summary_path = PROCESSED_DIR / "retrieval_summary.json"
    save_json(summaries, summary_path)

    print_header("Retrieval Summary Saved")
    print(f"Saved retrieval summary to: {summary_path}")

if __name__ == "__main__":
    main()