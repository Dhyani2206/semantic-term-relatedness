from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_CSV_PATH,
    PROCESSED_DIR,
    RANDOM_SEED,
    SPLIT_SUMMARY_PATH,
    TEST_CSV_PATH,
    TEST_RATIO,
    TRAIN_CSV_PATH,
    TRAIN_RATIO,
    VAL_CSV_PATH,
    VAL_RATIO,
    ensure_directories,
)
from src.utils import print_header, save_json, series_distribution

def validate_required_columns(df: pd.DataFrame) -> None:
    required = {"topic", "term1", "term2", "score", "relatedness_level"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in cleaned dataset: {sorted(missing)}")

def summarize_split(df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    return {
        "split_name": split_name,
        "rows": int(len(df)),
        "unique_topics": int(df["topic"].nunique()),
        "unique_term1": int(df["term1"].nunique()),
        "unique_term2": int(df["term2"].nunique()),
        "score_min": float(df["score"].min()) if not df.empty else None,
        "score_max": float(df["score"].max()) if not df.empty else None,
        "score_mean": float(df["score"].mean()) if not df.empty else None,
        "score_std": float(df["score"].std()) if not df.empty else None,
        "score_median": float(df["score"].median()) if not df.empty else None,
        "score_q25": float(df["score"].quantile(0.25)) if not df.empty else None,
        "score_q75": float(df["score"].quantile(0.75)) if not df.empty else None,
        "relatedness_distribution": series_distribution(df["relatedness_level"], normalize=False),
    }

def assert_no_topic_overlap(
    train_topics: set[str],
    val_topics: set[str],
    test_topics: set[str],
) -> None:
    if not train_topics.isdisjoint(val_topics):
        raise AssertionError("Train and validation topics overlap.")
    if not train_topics.isdisjoint(test_topics):
        raise AssertionError("Train and test topics overlap.")
    if not val_topics.isdisjoint(test_topics):
        raise AssertionError("Validation and test topics overlap.")

def main() -> None:
    ensure_directories()

    print_header("Step 1: Load Clean Dataset")
    df = pd.read_csv(CLEAN_CSV_PATH)
    validate_required_columns(df)
    print(f"Loaded cleaned dataset shape: {df.shape}")

    print_header("Step 2: Extract Unique Topics")
    topics = sorted(df["topic"].unique())
    n_topics = len(topics)
    print(f"Total unique topics: {n_topics}")

    if n_topics < 3:
        raise ValueError("Need at least 3 topics to create train/val/test splits.")

    print_header("Step 3: Topic-wise Split")
    # First split off train
    train_topics, temp_topics = train_test_split(
        topics,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    # Split remaining topics into val and test proportionally
    remaining_ratio = VAL_RATIO + TEST_RATIO
    if remaining_ratio <= 0:
        raise ValueError("VAL_RATIO + TEST_RATIO must be greater than 0.")

    val_relative_ratio = VAL_RATIO / remaining_ratio

    val_topics, test_topics = train_test_split(
        temp_topics,
        train_size=val_relative_ratio,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_topic_set = set(train_topics)
    val_topic_set = set(val_topics)
    test_topic_set = set(test_topics)

    assert_no_topic_overlap(train_topic_set, val_topic_set, test_topic_set)

    print(f"Train topics: {len(train_topic_set)}")
    print(f"Val topics:   {len(val_topic_set)}")
    print(f"Test topics:  {len(test_topic_set)}")

    print_header("Step 4: Build Split DataFrames")
    train_df = df[df["topic"].isin(train_topic_set)].reset_index(drop=True)
    val_df = df[df["topic"].isin(val_topic_set)].reset_index(drop=True)
    test_df = df[df["topic"].isin(test_topic_set)].reset_index(drop=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    print_header("Step 5: Save Split Files")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_CSV_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_CSV_PATH, index=False, encoding="utf-8")

    summary = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "topic_counts": {
            "train": len(train_topic_set),
            "val": len(val_topic_set),
            "test": len(test_topic_set),
        },
        "topic_overlap": {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        },
        "train": summarize_split(train_df, "train"),
        "val": summarize_split(val_df, "val"),
        "test": summarize_split(test_df, "test"),
    }
    save_json(summary, SPLIT_SUMMARY_PATH)

    print(f"Saved train split to: {TRAIN_CSV_PATH}")
    print(f"Saved val split to:   {VAL_CSV_PATH}")
    print(f"Saved test split to:  {TEST_CSV_PATH}")
    print(f"Saved split summary to: {SPLIT_SUMMARY_PATH}")

    print_header("Split Summary")
    for split_name in ["train", "val", "test"]:
        split_info = summary[split_name]
        print(
            f"{split_name.upper()}: "
            f"rows={split_info['rows']}, "
            f"topics={split_info['unique_topics']}, "
            f"score_mean={split_info['score_mean']:.4f}"
        )

if __name__ == "__main__":
    main()