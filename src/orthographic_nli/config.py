from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    rng_seed: int
    requests_per_minute: int
    max_examples_per_condition: int
    dataset_dir: str
    eval_split: str
    languages: List[str]
    groq_api_keys: List[str]
    results_dir: str
    write_traces: bool


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        rng_seed=int(os.getenv("RNG_SEED", "13")),
        requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "60")),
        max_examples_per_condition=int(os.getenv("MAX_EXAMPLES_PER_CONDITION", "100")),
        dataset_dir=os.getenv("DATASET_DIR", "../input/xnli-multilingual-nli-dataset"),
        eval_split=os.getenv("EVAL_SPLIT", "test"),
        languages=_parse_list(os.getenv("LANGUAGES", "ar,ur,en,sw")),
        groq_api_keys=_parse_list(os.getenv("GROQ_API_KEYS", "")),
        results_dir=os.getenv("RESULTS_DIR", "./results"),
        write_traces=bool(int(os.getenv("WRITE_TRACES", "0"))),
    )
