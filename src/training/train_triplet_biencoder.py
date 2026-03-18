'''
train gte-small on triplets
- validate on validation triplets
- save fine-tuned checkpoints
- save training metadat for reproduciblitty
'''
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.config import (
    BIENCODER_MODELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    MODELS_DIR,
    RANDOM_SEED,
    TRIPLETS_TRAIN_PATH,
    TRIPLETS_VAL_PATH,
    TRIPLET_MARGIN,
    ensure_directories,
)
from src.utils import print_header, save_json, set_global_seed

MODEL_KEY = "gte_small"
OUTPUT_DIR = MODELS_DIR / "triplet_finetuned" / "gte_small_triplet"

@dataclass(frozen=True)
class TrainingConfig:
    model_key: str = MODEL_KEY
    batch_size: int = DEFAULT_BATCH_SIZE
    num_epochs: int = DEFAULT_NUM_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    triplet_margin: float = TRIPLET_MARGIN
    random_seed: int = RANDOM_SEED
    warmup_ratio: float = 0.1

def validate_triplet_columns(df: pd.DataFrame, split_name: str) -> None:
    required = {"anchor", "positive", "negative", "negative_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {split_name} triplets: {sorted(missing)}")

def load_triplet_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRIPLETS_TRAIN_PATH)
    val_df = pd.read_csv(TRIPLETS_VAL_PATH)

    validate_triplet_columns(train_df, "train")
    validate_triplet_columns(val_df, "val")

    return train_df, val_df

def convert_to_input_examples(df: pd.DataFrame) -> list[InputExample]:
    examples: list[InputExample] = []

    for _, row in df.iterrows():
        examples.append(
            InputExample(
                texts=[
                    str(row["anchor"]),
                    str(row["positive"]),
                    str(row["negative"]),
                ]
            )
        )
    return examples

def build_dataloader(examples: list[InputExample], batch_size: int) -> DataLoader:
    return DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
    )

def summarize_triplets(df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    summary = {
        "split_name": split_name,
        "num_triplets": int(len(df)),
        "num_unique_anchors": int(df["anchor"].nunique()),
        "num_unique_positives": int(df["positive"].nunique()),
        "num_unique_negatives": int(df["negative"].nunique()),
        "negative_type_distribution": df["negative_type"].value_counts(dropna=False).to_dict(),
    }
    return summary

def main() -> None:
    ensure_directories()
    set_global_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig()

    if config.model_key not in BIENCODER_MODELS:
        raise ValueError(
            f"Unknown model_key='{config.model_key}'. "
            f"Available keys: {sorted(BIENCODER_MODELS.keys())}"
        )

    base_model_name = BIENCODER_MODELS[config.model_key]

    print_header("Step 1: Load Triplet Data")
    train_df, val_df = load_triplet_data()
    print(f"Train triplets shape: {train_df.shape}")
    print(f"Val triplets shape:   {val_df.shape}")

    train_summary = summarize_triplets(train_df, "train")
    val_summary = summarize_triplets(val_df, "val")

    print_header("Step 2: Build Input Examples")
    train_examples = convert_to_input_examples(train_df)
    val_examples = convert_to_input_examples(val_df)
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}")

    print_header("Step 3: Load Base Model")
    model = SentenceTransformer(base_model_name)
    print(f"Loaded base model: {base_model_name}")

    print_header("Step 4: Prepare DataLoader and Loss")
    train_dataloader = build_dataloader(
        examples=train_examples,
        batch_size=config.batch_size,
    )

    train_loss = losses.TripletLoss(
        model=model,
        triplet_margin=config.triplet_margin,
    )

    warmup_steps = int(len(train_dataloader) * config.num_epochs * config.warmup_ratio)

    print(f"Batch size:    {config.batch_size}")
    print(f"Num epochs:    {config.num_epochs}")
    print(f"Warmup steps:  {warmup_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Triplet margin:{config.triplet_margin}")

    print_header("Step 5: Train Model")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=str(OUTPUT_DIR),
        show_progress_bar=True,
    )

    print_header("Step 6: Save Training Metadata")
    metadata = {
        "training_config": asdict(config),
        "base_model_name": base_model_name,
        "output_dir": str(OUTPUT_DIR),
        "train_summary": train_summary,
        "val_summary": val_summary,
    }

    metadata_path = OUTPUT_DIR / "training_metadata.json"
    save_json(metadata, metadata_path)

    print(f"Saved fine-tuned model to: {OUTPUT_DIR}")
    print(f"Saved metadata to:         {metadata_path}")

    print_header("Training Complete")
    print("Triplet fine-tuned bi-encoder is ready for retrieval evaluation.")

if __name__ == "__main__":
    main()