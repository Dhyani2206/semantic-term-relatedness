from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Raw dataset files
RAW_ZIP_FILENAME = "IBM_Debater_TR9856.zip"
RAW_CSV_FILENAME = "TermRelatednessResults.csv"

RAW_ZIP_PATH = RAW_DIR / RAW_ZIP_FILENAME
CLEAN_CSV_PATH = INTERIM_DIR / "clean.csv"
CLEAN_SUMMARY_PATH = INTERIM_DIR / "clean_summary.json"

TRAIN_CSV_PATH = PROCESSED_DIR / "mwtr_train.csv"
VAL_CSV_PATH = PROCESSED_DIR / "mwtr_val.csv"
TEST_CSV_PATH = PROCESSED_DIR / "mwtr_test.csv"
SPLIT_SUMMARY_PATH = PROCESSED_DIR / "split_summary.json"

RETRIEVAL_CORPUS_PATH = PROCESSED_DIR / "retrieval_corpus.csv"
RETRIEVAL_QRELS_TRAIN_PATH = PROCESSED_DIR / "retrieval_qrels_train.csv"
RETRIEVAL_QRELS_VAL_PATH = PROCESSED_DIR / "retrieval_qrels_val.csv"
RETRIEVAL_QRELS_TEST_PATH = PROCESSED_DIR / "retrieval_qrels_test.csv"

TRIPLETS_TRAIN_PATH = PROCESSED_DIR / "triplets_train.csv"
TRIPLETS_VAL_PATH = PROCESSED_DIR / "triplets_val.csv"

# dataset config
DATASET_URL = (
    "https://www.research.ibm.com/haifa/dept/vst/files/"
    "IBM_Debater_%28R%29_TR9856.v2.zip"
)

REQUIRED_COLUMNS = ["topic", "term1", "term2", "score"]
TEXT_COLUMNS = ["topic", "term1", "term2"]

SCORE_MIN = 0.0
SCORE_MAX = 1.0

# Auxiliary bins for analysis / reporting only
LOW_MAX = 0.33
MEDIUM_MAX = 0.66
RELATEDNESS_LABELS = ["low", "medium", "high"]

# Split config
RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")

# retrieval config
TOP_K_DEFAULT = 50
TOP_K_EVAL = [5, 10, 20, 50]

# Relevance threshold for retrieval qrels
RETRIEVAL_POSITIVE_THRESHOLD = 0.70

# score thresholds for triplet construction
TRIPLET_POSITIVE_THRESHOLD = 0.75
TRIPLET_HARD_NEGATIVE_MIN = 0.40
TRIPLET_HARD_NEGATIVE_MAX = 0.60
TRIPLET_NEGATIVE_THRESHOLD = 0.30

# Model config
BIENCODER_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "gte_small": "thenlper/gte-small",
}

CROSSENCODER_MODELS = {
    "minilm_cross": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "mpnet_cross": "cross-encoder/stsb-roberta-base",
}

LLM_RERANKER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

#Training config
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_LENGTH = 128

TRIPLET_MARGIN = 0.25

# Runtime Helpers
ALL_REQUIRED_DIRS = [
    DATA_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    EXTERNAL_DIR,
    NOTEBOOKS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    LOGS_DIR,
]

def ensure_directories() -> None:
    """Create all required project directories if they do not already exist."""
    for directory in ALL_REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO
    random_seed: int = RANDOM_SEED

@dataclass(frozen=True)
class RetrievalConfig:
    top_k_default: int = TOP_K_DEFAULT
    top_k_eval: tuple[int, ...] = tuple(TOP_K_EVAL)
    positive_threshold: float = RETRIEVAL_POSITIVE_THRESHOLD

@dataclass(frozen=True)
class TripletConfig:
    positive_threshold: float = TRIPLET_POSITIVE_THRESHOLD
    hard_negative_min: float = TRIPLET_HARD_NEGATIVE_MIN
    hard_negative_max: float = TRIPLET_HARD_NEGATIVE_MAX
    negative_threshold: float = TRIPLET_NEGATIVE_THRESHOLD
    margin: float = TRIPLET_MARGIN