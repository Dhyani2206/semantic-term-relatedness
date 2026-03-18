'''
triplets:
anchor = term 1
positive =  'term 2' high relatedness with anchor (score >= 0.75)
negative = 'term 2' with low relatedness with anchor (score <= 0.30)
prefer hard negatives when available. (0.40 <= score <= 0.60)

- one triplet per positive pair, and one selected negative per anchor-positive pair.
'''
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from src.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    TRAIN_CSV_PATH,
    TRIPLETS_TRAIN_PATH,
    TRIPLETS_VAL_PATH,
    TRIPLET_HARD_NEGATIVE_MAX,
    TRIPLET_HARD_NEGATIVE_MIN,
    TRIPLET_NEGATIVE_THRESHOLD,
    TRIPLET_POSITIVE_THRESHOLD,
    VAL_CSV_PATH,
    ensure_directories,
)

from src.utils import print_header, save_json, set_global_seed
def validate_split_columns(df: pd.DataFrame, split_name: str) -> None:
    required = {"topic", "term1", "term2", "score", "relatedness_level"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {split_name} split: {sorted(missing)}"
        )

def choose_negative(
    candidates_hard: list[str],
    candidates_easy: list[str],
    rng: np.random.Generator,
) -> tuple[str | None, str | None]:
    """
    Choose a negative term, preferring hard negatives when available.
    Returns:
        (negative_text, negative_type)
        negative_type is one of: "hard", "easy", None
    """
    if candidates_hard:
        idx = int(rng.integers(0, len(candidates_hard)))
        return candidates_hard[idx], "hard"

    if candidates_easy:
        idx = int(rng.integers(0, len(candidates_easy)))
        return candidates_easy[idx], "easy"
    return None, None

def build_triplets_for_split(
    df: pd.DataFrame,
    split_name: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build triplets from a split DataFrame.

    Triplet logic:
    - anchor = term1
    - positive = term2 where score >= positive threshold
    - hard negative = term2 where score is in [hard_negative_min, hard_negative_max]
    - easy negative = term2 where score <= negative threshold

    One triplet is generated per positive pair when at least one valid negative exists.
    """
    validate_split_columns(df, split_name)

    triplets: list[dict[str, Any]] = []

    unique_anchors = sorted(df["term1"].unique())
    global_term2_pool = sorted(df["term2"].drop_duplicates().tolist())
    global_negative_triplets = 0
    anchors_with_positive = 0
    anchors_without_positive = 0
    anchors_with_triplets = 0
    anchors_without_negatives = 0

    hard_negative_triplets = 0
    easy_negative_triplets = 0

    for anchor in unique_anchors:
        anchor_rows = df[df["term1"] == anchor].copy()

        positives = (
            anchor_rows[anchor_rows["score"] >= TRIPLET_POSITIVE_THRESHOLD]["term2"]
            .drop_duplicates()
            .tolist()
        )

        hard_negatives = (
            anchor_rows[
                (anchor_rows["score"] >= TRIPLET_HARD_NEGATIVE_MIN)
                & (anchor_rows["score"] <= TRIPLET_HARD_NEGATIVE_MAX)
            ]["term2"]
            .drop_duplicates()
            .tolist()
        )

        easy_negatives = (
            anchor_rows[anchor_rows["score"] <= TRIPLET_NEGATIVE_THRESHOLD]["term2"]
            .drop_duplicates()
            .tolist()
        )

        # Prevent accidental overlap between positive and negative pools
        positive_set = set(positives)
        hard_negatives = [x for x in hard_negatives if x not in positive_set]
        easy_negatives = [x for x in easy_negatives if x not in positive_set]

        if not positives:
            anchors_without_positive += 1
            continue

        anchors_with_positive += 1
        anchor_generated_any = False

        for positive in positives:
            negative, negative_type = choose_negative(
                candidates_hard=hard_negatives,
                candidates_easy=easy_negatives,
                rng=rng,
            )

            if negative is None:
                invalid_choices = positive_set.union({anchor})
                global_negatives = [
                    term for term in global_term2_pool
                    if term not in invalid_choices
                ]

                if global_negatives:
                    idx = int(rng.integers(0, len(global_negatives)))
                    negative = global_negatives[idx]
                    negative_type = "global_easy"
                else:
                    continue

            triplets.append(
                {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative,
                    "negative_type": negative_type,
                }
            )

            anchor_generated_any = True

            if negative_type == "hard":
                hard_negative_triplets += 1
            elif negative_type == "easy":
                easy_negative_triplets += 1
            elif negative_type == "global_easy":
                global_negative_triplets += 1

        if anchor_generated_any:
            anchors_with_triplets += 1
        else:
            anchors_without_negatives += 1

    triplets_df = pd.DataFrame(triplets)

    summary = {
        "split_name": split_name,
        "input_rows": int(len(df)),
        "unique_anchors": int(len(unique_anchors)),
        "anchors_with_positive": int(anchors_with_positive),
        "anchors_without_positive": int(anchors_without_positive),
        "anchors_with_triplets": int(anchors_with_triplets),
        "anchors_without_valid_negatives": int(anchors_without_negatives),
        "triplet_count": int(len(triplets_df)),
        "hard_negative_triplets": int(hard_negative_triplets),
        "easy_negative_triplets": int(easy_negative_triplets),
        "global_negative_triplets": int(global_negative_triplets),
    }

    return triplets_df, summary

def main() -> None:
    ensure_directories()
    set_global_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    print_header("Step 1: Load Train/Val Splits")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")

    print_header("Step 2: Build Train Triplets")
    train_triplets, train_summary = build_triplets_for_split(
        df=train_df,
        split_name="train",
        rng=rng,
    )
    print(f"Train triplets shape: {train_triplets.shape}")

    print_header("Step 3: Build Val Triplets")
    val_triplets, val_summary = build_triplets_for_split(
        df=val_df,
        split_name="val",
        rng=rng,
    )
    print(f"Val triplets shape: {val_triplets.shape}")

    print_header("Step 4: Save Triplet Files")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_triplets.to_csv(TRIPLETS_TRAIN_PATH, index=False, encoding="utf-8")
    val_triplets.to_csv(TRIPLETS_VAL_PATH, index=False, encoding="utf-8")

    summary = {
        "random_seed": RANDOM_SEED,
        "thresholds": {
            "positive_threshold": TRIPLET_POSITIVE_THRESHOLD,
            "hard_negative_min": TRIPLET_HARD_NEGATIVE_MIN,
            "hard_negative_max": TRIPLET_HARD_NEGATIVE_MAX,
            "negative_threshold": TRIPLET_NEGATIVE_THRESHOLD,
        },
        "train": train_summary,
        "val": val_summary,
    }
    summary_path = PROCESSED_DIR / "triplets_summary.json"
    save_json(summary, summary_path)

    print(f"Saved train triplets to: {TRIPLETS_TRAIN_PATH}")
    print(f"Saved val triplets to:   {TRIPLETS_VAL_PATH}")
    print(f"Saved triplet summary to: {summary_path}")

    print_header("Triplet Summary")
    for split_name in ["train", "val"]:
        split_info = summary[split_name]
        print(
            f"{split_name.upper()}: "
            f"triplets={split_info['triplet_count']}, "
            f"anchors_with_positive={split_info['anchors_with_positive']}, "
            f"hard_negatives={split_info['hard_negative_triplets']}, "
            f"easy_negatives={split_info['easy_negative_triplets']}"
        )

if __name__ == "__main__":
    main()