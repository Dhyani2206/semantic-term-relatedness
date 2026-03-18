'''
- load test split
- evaluate embedding models
- encode term1 and term2
- compute cosine similarity
- compute pearson, spearman
- save result table.

it  gives benchmark for later retrieval comparison
'''
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import BIENCODER_MODELS, TABLES_DIR, TEST_CSV_PATH, ensure_directories
from src.utils import print_header, save_json


def validate_columns(df: pd.DataFrame) -> None:
    required = {"term1", "term2", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

def compute_cosine_scores(
    model: SentenceTransformer,
    term1_list: list[str],
    term2_list: list[str],
) -> np.ndarray:
    """
    Encode paired terms and compute cosine similarity row-wise.
    """
    emb1 = model.encode(term1_list, convert_to_numpy=True, show_progress_bar=True)
    emb2 = model.encode(term2_list, convert_to_numpy=True, show_progress_bar=True)
    sims = cosine_similarity(emb1, emb2)
    rowwise_scores = np.diag(sims)
    return rowwise_scores

def evaluate_model(
    model_name: str,
    model_path: str,
    df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Evaluate one embedding model on pairwise semantic similarity.
    """
    print_header(f"Evaluating {model_name}")
    model = SentenceTransformer(model_path)

    predictions = compute_cosine_scores(
        model=model,
        term1_list=df["term1"].tolist(),
        term2_list=df["term2"].tolist(),
    )

    gold = df["score"].to_numpy()
    pearson_value, _ = pearsonr(gold, predictions)
    spearman_value, _ = spearmanr(gold, predictions)

    result = {
        "model_name": model_name,
        "model_path": model_path,
        "num_examples": int(len(df)),
        "pearson": float(pearson_value),
        "spearman": float(spearman_value),
        "pred_mean": float(np.mean(predictions)),
        "pred_std": float(np.std(predictions)),
        "gold_mean": float(np.mean(gold)),
        "gold_std": float(np.std(gold)),
    }

    pred_df = df.copy()
    pred_df["predicted_similarity"] = predictions
    pred_df["model_name"] = model_name

    return result, pred_df

def main() -> None:
    ensure_directories()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print_header("Load Test Split")
    df = pd.read_csv(TEST_CSV_PATH)
    validate_columns(df)
    print(f"Test shape: {df.shape}")

    all_results: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []

    for model_name, model_path in BIENCODER_MODELS.items():
        result, pred_df = evaluate_model(
            model_name=model_name,
            model_path=model_path,
            df=df,
        )
        all_results.append(result)
        all_predictions.append(pred_df)

    results_df = pd.DataFrame(all_results).sort_values(
        by=["spearman", "pearson"],
        ascending=False,
    ).reset_index(drop=True)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    results_path = TABLES_DIR / "baseline_embedding_results.csv"
    predictions_path = TABLES_DIR / "baseline_embedding_predictions.csv"
    summary_path = TABLES_DIR / "baseline_embedding_results.json"

    results_df.to_csv(results_path, index=False, encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")
    save_json(
        {
            "results": results_df.to_dict(orient="records"),
            "best_model_by_spearman": results_df.iloc[0]["model_name"] if not results_df.empty else None,
        },
        summary_path,
    )
    print_header("Baseline Embedding Results")
    print(results_df)

    print(f"Saved results table to: {results_path}")
    print(f"Saved predictions to:  {predictions_path}")
    print(f"Saved summary to:      {summary_path}")

if __name__ == "__main__":
    main()