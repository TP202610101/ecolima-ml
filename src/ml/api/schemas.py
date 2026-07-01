"""Pydantic schemas for the EcoLima ML API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ml.config import DEFAULT_SERVING_MODEL


class PredictionRequest(BaseModel):
    zone: dict[str, Any]
    zone_id: str | int | None = None
    model_name: str = DEFAULT_SERVING_MODEL
    include_explanation: bool = False


class BatchPredictionRequest(BaseModel):
    zones: list[dict[str, Any]] = Field(min_length=1)
    zone_ids: list[str | int] | None = None
    model_name: str = DEFAULT_SERVING_MODEL


class RecommendationRequest(BatchPredictionRequest):
    top_n: int = Field(default=10, ge=1, le=500)
