"""Prepare the provisional real candidate dataset for EcoLima ML."""

from __future__ import annotations

import argparse
import json

from ml.dataset_builder import prepare_real_candidate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real provisional candidate dataset")
    parser.add_argument("--input", default=None, help="Optional source CSV path")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = prepare_real_candidate_dataset(input_path=args.input, output_path=args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
