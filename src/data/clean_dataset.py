from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    CLEAN_CSV_PATH,
    CLEAN_SUMMARY_PATH,
    RAW_CSV_FILENAME,
    RAW_DIR,
    REQUIRED_COLUMNS,
    RELATEDNESS_LABELS,
    SCORE_MAX,
    SCORE_MIN,
    TEXT_COLUMNS,
    ensure_directories,
)
from src.utils import print_header, save_json, series_distribution

def find_raw_csv(raw_dir: Path, target_filename: str = RAW_CSV_FILENAME) -> Path:
    """
    Locate the raw CSV file recursively inside the raw data directory.

    Raises:
        FileNotFoundError: If no matching CSV is found.
    """
    matches = list(raw_dir.rglob(target_filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{target_filename}' anywhere under: {raw_dir}"
        )

    if len(matches) > 1:
        print(
            f"Warning: multiple matches found for '{target_filename}'. "
            f"Using the first match: {matches[0]}"
        )

    return matches[0]

def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Ensure all required columns are present.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def normalize_text(text: Any) -> str:
    """
    Normalize text for consistent downstream matching.

    Steps:
    - convert to string
    - unicode normalize
    - lowercase
    - trim surrounding whitespace
    - collapse repeated internal whitespace
    """
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def add_relatedness_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add coarse relatedness labels for analysis/reporting.

    Note:
        These labels are auxiliary and should not replace the continuous
        score for main training objectives.
    """
    bins = [0.0, 0.33, 0.66, 1.0]
    df["relatedness_level"] = pd.cut(
        df["score"],
        bins=bins,
        labels=RELATEDNESS_LABELS,
        include_lowest=True,
    )
    return df

def validate_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce score to numeric and ensure all values lie in the valid range.
    """
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    non_numeric_count = int(df["score"].isna().sum())
    if non_numeric_count > 0:
        raise ValueError(
            f"Found {non_numeric_count} rows where 'score' could not be parsed as numeric."
        )

    out_of_range_mask = (df["score"] < SCORE_MIN) | (df["score"] > SCORE_MAX)
    out_of_range_count = int(out_of_range_mask.sum())
    if out_of_range_count > 0:
        bad_rows = df.loc[out_of_range_mask, ["topic", "term1", "term2", "score"]].head(5)
        raise ValueError(
            "Found score values outside the valid range "
            f"[{SCORE_MIN}, {SCORE_MAX}]. Example rows:\n{bad_rows}"
        )

    return df


def check_duplicate_score_conflicts(df: pd.DataFrame) -> int:
    """
    Check whether the same normalized (topic, term1, term2) pair appears
    with multiple distinct scores.

    Returns:
        Number of conflicting groups.
    """
    conflicts = (
        df.groupby(["topic", "term1", "term2"])["score"]
        .nunique()
        .gt(1)
        .sum()
    )
    return int(conflicts)

def count_reverse_pairs(df: pd.DataFrame) -> int:
    """
    Count how many rows have a reverse-pair counterpart:
    (topic, term1, term2) and (topic, term2, term1)

    This is not necessarily an error, but it is important to know for
    retrieval and triplet construction.
    """
    forward = df[["topic", "term1", "term2"]].copy()
    reverse = df[["topic", "term1", "term2"]].rename(
        columns={"term1": "term2", "term2": "term1"}
    )

    matched = forward.merge(reverse, on=["topic", "term1", "term2"], how="inner")
    return int(len(matched))

def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    - cleaned dataframe
    - summary dictionary
    """
    summary: dict[str, Any] = {}
    summary["rows_initial"] = int(len(df))

    validate_required_columns(df, REQUIRED_COLUMNS)
    df = df[REQUIRED_COLUMNS].copy()

    missing_before_drop = df[REQUIRED_COLUMNS].isna().sum().to_dict()
    summary["missing_values_before_drop"] = {k: int(v) for k, v in missing_before_drop.items()}

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    summary["rows_after_dropna"] = int(len(df))
    summary["rows_removed_due_to_missing"] = summary["rows_initial"] - summary["rows_after_dropna"]

    df = validate_score_column(df)

    for col in TEXT_COLUMNS:
        df[col] = df[col].map(normalize_text)

    empty_text_mask = (
        (df["topic"] == "") |
        (df["term1"] == "") |
        (df["term2"] == "")
    )
    empty_text_count = int(empty_text_mask.sum())
    summary["rows_with_empty_text_after_normalization"] = empty_text_count

    if empty_text_count > 0:
        df = df.loc[~empty_text_mask].copy()

    # Detect conflicts BEFORE fixing
    duplicate_conflicts = check_duplicate_score_conflicts(df)
    summary["duplicate_pair_score_conflicts"] = duplicate_conflicts

    # Count duplicate groups (for reporting)
    pair_group_sizes = (
        df.groupby(["topic", "term1", "term2"])
        .size()
        .reset_index(name="count")
    )
    summary["duplicate_pair_groups"] = int((pair_group_sizes["count"] > 1).sum())

    df = (
        df.groupby(["topic", "term1", "term2"], as_index=False)
        .agg(score=("score", "mean"))
        .reset_index(drop=True)
    )

    summary["duplicate_pairs_resolved_via_mean"] = summary["duplicate_pair_groups"]

    df = df.drop_duplicates(subset=["topic", "term1", "term2"]).reset_index(drop=True)
    reverse_pair_count = count_reverse_pairs(df)
    summary["reverse_pair_matches"] = reverse_pair_count
    df = add_relatedness_level(df)

    summary["rows_final"] = int(len(df))
    summary["unique_topics"] = int(df["topic"].nunique())
    summary["unique_term1"] = int(df["term1"].nunique())
    summary["unique_term2"] = int(df["term2"].nunique())
    summary["score_min"] = float(df["score"].min())
    summary["score_max"] = float(df["score"].max())
    summary["score_mean"] = float(df["score"].mean())
    summary["score_std"] = float(df["score"].std())
    summary["relatedness_distribution"] = series_distribution(df["relatedness_level"], normalize=False)

    return df, summary

def main() -> None:
    ensure_directories()

    print_header("Step 1: Locate Raw CSV")
    raw_csv_path = find_raw_csv(RAW_DIR)
    print(f"Raw CSV found at: {raw_csv_path}")

    print_header("Step 2: Load Raw Dataset")
    df = pd.read_csv(raw_csv_path, encoding="latin-1")
    print(f"Loaded dataframe shape: {df.shape}")

    print_header("Step 3: Clean Dataset")
    clean_df, summary = clean_dataset(df)
    print(f"Cleaned dataframe shape: {clean_df.shape}")

    print_header("Step 4: Save Outputs")
    CLEAN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_CSV_PATH, index=False, encoding="utf-8")
    save_json(summary, CLEAN_SUMMARY_PATH)

    print(f"Saved cleaned dataset to: {CLEAN_CSV_PATH}")
    print(f"Saved cleaning summary to: {CLEAN_SUMMARY_PATH}")
    print_header("Cleaning Summary")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()