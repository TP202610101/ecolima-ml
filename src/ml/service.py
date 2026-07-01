"""Service layer consumed by CLI scripts or FastAPI endpoints."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Any

from ml.config import DEFAULT_SERVING_MODEL, MODELS_DIR
from ml.predictor import RecyclingPredictor


class ModelService:
    def __init__(self, model_name: str = DEFAULT_SERVING_MODEL):
        self.model_name = model_name
        self.predictor = RecyclingPredictor.load(model_name)

    def health(self) -> dict[str, Any]:
        model_path = MODELS_DIR / f"{self.model_name}.joblib"
        return {
            "status": "ok",
            "model_name": self.model_name,
            "model_loaded": True,
            "model_path": str(model_path),
            "threshold": self.predictor.threshold,
            "target": self.predictor.metadata.get("target"),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "threshold": self.predictor.threshold,
            "metadata": self.predictor.metadata,
        }

    def predict_records(
        self,
        records: list[dict[str, Any]],
        zone_ids: list[str | int] | None = None,
    ) -> list[dict[str, Any]]:
        results = self.predictor.predict(records, zone_ids=zone_ids)
        return [asdict(result) for result in results]

    def explain_record(self, record: dict[str, Any], zone_id: str | int = 0) -> dict[str, Any]:
        return asdict(self.predictor.predict_with_explanation(record, zone_id=zone_id))

    def rank_records(
        self,
        records: list[dict[str, Any]],
        zone_ids: list[str | int] | None = None,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        results = self.predictor.rank_zones(records, zone_ids=zone_ids, top_n_zones=top_n)
        return [asdict(result) for result in results]


@lru_cache(maxsize=4)
def get_model_service(model_name: str = DEFAULT_SERVING_MODEL) -> ModelService:
    return ModelService(model_name=model_name)
