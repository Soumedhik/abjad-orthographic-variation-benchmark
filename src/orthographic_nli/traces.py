from __future__ import annotations

import json
import time
from typing import List

import pandas as pd

from .groq_client import ModelSpec, build_key_cycle, run_model


def log_traces(
    df: pd.DataFrame,
    specs: List[ModelSpec],
    groq_keys: List[str],
    requests_per_minute: int,
    output_path: str,
    rng_seed: int,
    per_condition: int = 20,
) -> None:
    key_cycle = build_key_cycle(groq_keys)
    sleep_between_calls = 60 / max(requests_per_minute, 1)
    with open(output_path, "w", encoding="utf-8") as handle:
        grouped = df.groupby(["language", "condition"])
        for (lang, cond), subset in grouped:
            subset = subset.sample(min(per_condition, len(subset)), random_state=rng_seed)
            for spec in specs:
                for _, row in subset.iterrows():
                    pred = run_model(spec, row.premise, row.hypothesis, key_cycle)
                    record = {
                        "provider": spec.provider,
                        "model": spec.model,
                        "language": lang,
                        "condition": cond,
                        "label": row.label,
                        "prediction": pred,
                        "premise": row.premise,
                        "hypothesis": row.hypothesis,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    time.sleep(sleep_between_calls)
