from __future__ import annotations

import json
import time
from typing import List, Tuple

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from tqdm.auto import tqdm

from .groq_client import LABEL_ORDER, ModelSpec, build_key_cycle, run_model


def evaluate(
    df: pd.DataFrame,
    specs: List[ModelSpec],
    groq_keys: List[str],
    requests_per_minute: int,
    max_examples_per_condition: int,
    rng_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate multiple models across all orthographic conditions.
    
    Args:
        df: DataFrame containing premise-hypothesis pairs with language and condition.
        specs: List of model specifications to evaluate.
        groq_keys: API keys for Groq inference.
        requests_per_minute: Rate limit for API calls.
        max_examples_per_condition: Maximum examples to evaluate per condition.
        rng_seed: Random seed for reproducible sampling.
        
    Returns:
        Tuple of (results_df, predictions_df):
            - results_df: Accuracy, F1, and confusion matrix per model-language-condition.
            - predictions_df: Per-example predictions with metadata.
    """
    results = []
    predictions = []
    key_cycle = build_key_cycle(groq_keys)
    sleep_between_calls = 60 / max(requests_per_minute, 1)
    grouped = df.groupby(["language", "condition"])

    for (lang, cond), subset in grouped:
        if subset.empty:
            continue
        subset = subset.sample(min(max_examples_per_condition, len(subset)), random_state=rng_seed)
        for spec in specs:
            truths: List[str] = []
            preds: List[str] = []
            for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"{spec.model} {lang} {cond}"):
                pred = run_model(spec, row.premise, row.hypothesis, key_cycle)
                preds.append(pred)
                truths.append(row.label)
                predictions.append({
                    "provider": spec.provider,
                    "model": spec.model,
                    "language": lang,
                    "condition": cond,
                    "premise": row.premise,
                    "hypothesis": row.hypothesis,
                    "label": row.label,
                    "prediction": pred,
                })
                time.sleep(sleep_between_calls)
            acc = sum(p == t for p, t in zip(preds, truths)) / len(subset)
            macro_f1 = f1_score(truths, preds, labels=LABEL_ORDER, average="macro", zero_division=0)
            cm = confusion_matrix(truths, preds, labels=LABEL_ORDER)
            results.append({
                "provider": spec.provider,
                "model": spec.model,
                "language": lang,
                "condition": cond,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "examples": len(subset),
                "confusion_matrix": json.dumps(cm.tolist()),
            })
    return pd.DataFrame(results), pd.DataFrame(predictions)
