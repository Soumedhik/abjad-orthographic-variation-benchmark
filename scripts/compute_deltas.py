from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from orthographic_nli.metrics import compute_deltas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute robustness deltas from benchmark CSV.")
    parser.add_argument("--benchmark", type=str, required=True, help="Path to benchmark.csv")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench_path = Path(args.benchmark)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(bench_path)
    delta_acc = compute_deltas(df, metric="accuracy")
    delta_f1 = compute_deltas(df, metric="macro_f1")

    delta_acc.to_csv(out_dir / "robustness_deltas_accuracy.csv", index=False)
    delta_f1.to_csv(out_dir / "robustness_deltas_f1.csv", index=False)

    print(f"Wrote deltas to {out_dir}")


if __name__ == "__main__":
    main()
