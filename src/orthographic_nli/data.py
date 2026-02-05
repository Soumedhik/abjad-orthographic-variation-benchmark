from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


def _normalize_label(val) -> str:
    if isinstance(val, (int, np.integer)):
        return LABEL_MAP.get(int(val), str(val))
    try:
        as_int = int(val)
        return LABEL_MAP.get(as_int, str(val))
    except Exception:
        return str(val).strip().lower()


def load_local_xnli(dataset_dir: Path, language: str, split: str, limit: Optional[int] = None) -> pd.DataFrame:
    path = dataset_dir / f"{language}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path)
    required = {"premise", "hypothesis", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {required - set(df.columns)}")
    if limit:
        df = df.head(limit)
    df = df.copy()
    df["label_text"] = df["label"].apply(_normalize_label)
    df["language"] = language
    return df[["premise", "hypothesis", "label", "label_text", "language"]]
