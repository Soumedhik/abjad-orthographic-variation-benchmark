from __future__ import annotations

import pandas as pd


def compute_deltas(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Calculate performance degradation relative to clean baseline.
    
    Args:
        df: Benchmark results with accuracy or F1 scores per condition.
        metric: Name of metric column (e.g., 'accuracy' or 'macro_f1').
        
    Returns:
        DataFrame with baseline performance and delta columns showing degradation.
    """
    pivot = df.pivot_table(index=["provider", "model", "language"], columns="condition", values=metric)
    clean = pivot.get("clean")
    drops = {}
    for col in pivot.columns:
        if col == "clean":
            continue
        drops[f"drop_{metric}_{col}"] = clean - pivot[col]
    out = pd.concat([pivot, pd.DataFrame(drops)], axis=1).reset_index()
    return out
