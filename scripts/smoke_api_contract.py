"""Smoke test for the service contract without running FastAPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.config import DEFAULT_SERVING_MODEL, DEFAULT_SIMULATED_TRAINING_DATASET, API_OUTPUTS_DIR
from ml.service import get_model_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test EcoLima ML service contract")
    parser.add_argument("--dataset", default=str(DEFAULT_SIMULATED_TRAINING_DATASET))
    parser.add_argument("--model-name", default=DEFAULT_SERVING_MODEL)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--output", default=str(API_OUTPUTS_DIR / "smoke_api_contract.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset).head(args.n)
    records = df.to_dict(orient="records")
    ids = [r.get("candidate_id", i) for i, r in enumerate(records)]
    service = get_model_service(args.model_name)
    payload = {
        "health": service.health(),
        "metadata": service.metadata(),
        "predictions": service.predict_records(records, zone_ids=ids),
        "recommendations": service.rank_records(records, zone_ids=ids, top_n=min(args.n, 3)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[smoke_api_contract] Output saved: {output}")


if __name__ == "__main__":
    main()
