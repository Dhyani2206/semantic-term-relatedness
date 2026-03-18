from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def set_global_seed(seed: int) -> None:
    """Set Python and NumPy random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary to JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))

def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")

def dataframe_basic_summary(df: pd.DataFrame, name: str = "dataframe") -> dict[str, Any]:
    """Return a compact dataframe summary for logging/reporting."""
    summary = {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
    }
    return summary

def series_distribution(series: pd.Series, normalize: bool = False) -> dict[str, Any]:
    """Return value counts as a plain dictionary."""
    counts = series.value_counts(normalize=normalize, dropna=False)
    return {str(k): float(v) if normalize else int(v) for k, v in counts.items()}

def safe_mean(series: pd.Series) -> float | None:
    """Return float mean if series is non-empty, else None."""
    if series.empty:
        return None
    return float(series.mean())

def safe_std(series: pd.Series) -> float | None:
    """Return float std if series is non-empty, else None."""
    if series.empty:
        return None
    value = series.std()
    return None if pd.isna(value) else float(value)