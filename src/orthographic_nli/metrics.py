from __future__ import annotations

import pandas as pd


def compute_deltas(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = df.pivot_table(index=["provider", "model", "language"], columns="condition", values=metric)
    clean = pivot.get("clean")
    drops = {}
    for col in pivot.columns:
        if col == "clean":
            continue
        drops[f"drop_{metric}_{col}"] = clean - pivot[col]
    out = pd.concat([pivot, pd.DataFrame(drops)], axis=1).reset_index()
    return out
