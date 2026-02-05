from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from orthographic_nli.config import load_settings
from orthographic_nli.data import load_local_xnli
from orthographic_nli.evaluate import evaluate
from orthographic_nli.groq_client import ModelSpec
from orthographic_nli.traces import log_traces
from orthographic_nli.variants import build_token_pool, make_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run orthographic robustness benchmark.")
    parser.add_argument("--dataset-dir", type=str, help="Path to local XNLI CSV shards")
    parser.add_argument("--results-dir", type=str, help="Output directory for CSVs")
    parser.add_argument("--eval-split", type=str, help="XNLI split: train|validation|test")
    parser.add_argument("--languages", type=str, help="Comma-separated language list")
    parser.add_argument("--max-examples", type=int, help="Max examples per condition")
    parser.add_argument("--requests-per-minute", type=int, help="API rate limit")
    parser.add_argument("--write-traces", action="store_true", help="Write per-example traces")
    return parser.parse_args()


def main() -> None:
    settings = load_settings()
    args = parse_args()

    dataset_dir = Path(args.dataset_dir or settings.dataset_dir).expanduser()
    results_dir = Path(args.results_dir or settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_split = args.eval_split or settings.eval_split
    languages = [lang.strip() for lang in (args.languages or ",".join(settings.languages)).split(",") if lang.strip()]
    max_examples = args.max_examples or settings.max_examples_per_condition
    rpm = args.requests_per_minute or settings.requests_per_minute
    write_traces = args.write_traces or settings.write_traces

    random.seed(settings.rng_seed)
    np.random.seed(settings.rng_seed)

    frames = [load_local_xnli(dataset_dir, lang, eval_split) for lang in languages]
    base_df = pd.concat(frames, ignore_index=True)

    en_pool = build_token_pool(base_df[base_df.language == "en"].premise.tolist() + base_df[base_df.language == "en"].hypothesis.tolist())
    ur_pool = build_token_pool(base_df[base_df.language == "ur"].premise.tolist() + base_df[base_df.language == "ur"].hypothesis.tolist())

    rng = random.Random(settings.rng_seed)
    variants_df = make_variants(base_df, en_pool, ur_pool, rng)

    specs = [
        ModelSpec(provider="groq", model="llama-3.3-70b-versatile"),
        ModelSpec(provider="groq", model="llama-3.1-8b-instant"),
    ]

    results_df, predictions_df = evaluate(
        variants_df,
        specs,
        settings.groq_api_keys,
        rpm,
        max_examples,
        settings.rng_seed,
    )

    results_df.to_csv(results_dir / "benchmark.csv", index=False)
    predictions_df.to_csv(results_dir / "predictions_samples.csv", index=False)

    if write_traces:
        log_traces(
            variants_df,
            specs,
            settings.groq_api_keys,
            rpm,
            str(results_dir / "traces.jsonl"),
            settings.rng_seed,
        )

    print(f"Saved results to {results_dir}")


if __name__ == "__main__":
    main()
