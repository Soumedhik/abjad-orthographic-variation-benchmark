# Beyond Standardized Benchmarks: Quantifying LLM Degradation Under Realistic Orthographic Variation in Abjad Languages

This repository contains the code for evaluating LLM robustness to orthographic variation on XNLI splits. It generates controlled variants (diacritics removal, partial diacritics, romanization, mixed-script code-switching) and evaluates models via the Groq API.

## Quick start

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure your Groq key(s):

   ```bash
   copy .env.example .env
   ```

   Then edit `.env` and set `GROQ_API_KEYS` (comma-separated).

4. Run the benchmark:

   ```bash
   python scripts/run_benchmark.py --dataset-dir /path/to/xnli --results-dir ./results
   ```

## Expected dataset format

The loader expects local XNLI CSV shards named like:

```
<lang>_<split>.csv
```

Each file must include columns: `premise`, `hypothesis`, `label`.

Example:

```
ar_test.csv
ur_test.csv
en_test.csv
sw_test.csv
```

## Outputs

The benchmark writes:

- `benchmark.csv` with accuracy, macro F1, and confusion matrices.
- `predictions_samples.csv` with per-example predictions (optional).
- `robustness_deltas_accuracy.csv` and `robustness_deltas_f1.csv` from `compute_deltas.py`.

## Configuration

All settings can be passed as CLI flags or environment variables. See `.env.example` for defaults.

## Notes

- This code uses deterministic sampling via `RNG_SEED`.
- API rate limiting is controlled by `REQUESTS_PER_MINUTE`.
- No results or figures are included in this repository.
