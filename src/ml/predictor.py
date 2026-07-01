"""Inference wrapper for serialized EcoLima LightGBM models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from ml.config import ALL_FEATURES, MODELS_DIR, SHAP_TOP_N_FEATURES
from ml.explainer import explain_batch_from_df


@dataclass
class ZoneInput:
    population_density: float
    num_households: int
    nse_ab_pct: float
    nse_c_pct: float
    nse_de_pct: float
    urbanization_rate: float
    dist_nearest_road_m: float
    walkability_score: float
    poi_commercial_500m: int
    poi_educational_500m: int
    poi_parks_500m: int
    land_use_encoded: int
    area_m2: float
    dist_nearest_recycling_m: float
    recycling_density_1km: int
    waste_per_capita_kg: float
    coverage_gap_index: float
    accessibility_composite: float | None = None
    nse_high_ratio: float | None = None
    recycling_deficit: int | None = None


@dataclass
class PredictionResult:
    zone_id: str | int
    suitability_score: float
    is_recommended: bool
    top_features: list[dict] = field(default_factory=list)


class RecyclingPredictor:
    """Single-model inference facade used by scripts, API and backend adapters."""

    def __init__(self, model, preprocessor, feature_engineer, feature_names, metadata):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.feature_names = feature_names
        self.metadata = metadata
        self.threshold = float(metadata.get("serving_threshold", metadata.get("threshold", 0.5)))

    @classmethod
    def load(cls, model_name: str = "lgbm_recycling") -> "RecyclingPredictor":
        model_path = MODELS_DIR / f"{model_name}.joblib"
        meta_path = MODELS_DIR / f"{model_name}_metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        artifacts = joblib.load(model_path)
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        return cls(
            model=artifacts["model"],
            preprocessor=artifacts["preprocessor"],
            feature_engineer=artifacts["feature_engineer"],
            feature_names=artifacts.get("feature_names", ALL_FEATURES),
            metadata=metadata,
        )

    def _zone_to_df(self, zone: ZoneInput | dict) -> pd.DataFrame:
        if isinstance(zone, dict):
            data = {k: v for k, v in zone.items() if v is not None}
        else:
            data = {k: v for k, v in zone.__dict__.items() if v is not None}
        return pd.DataFrame([data])

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        X_eng = self.feature_engineer.transform(df)
        X_pre = self.preprocessor.transform(X_eng)
        return pd.DataFrame(X_pre, columns=self.feature_names)

    def predict(
        self,
        zones: list[ZoneInput | dict],
        zone_ids: list[str | int] | None = None,
    ) -> list[PredictionResult]:
        df = pd.concat([self._zone_to_df(z) for z in zones], ignore_index=True)
        X_pre = self._preprocess(df)
        probs = self.model.predict_proba(X_pre)[:, 1]
        ids = zone_ids or list(range(len(zones)))
        return [
            PredictionResult(
                zone_id=ids[i],
                suitability_score=round(float(probs[i]), 4),
                is_recommended=bool(probs[i] >= self.threshold),
            )
            for i in range(len(zones))
        ]

    def predict_with_explanation(
        self,
        zone: ZoneInput | dict,
        zone_id: str | int = 0,
        top_n: int = SHAP_TOP_N_FEATURES,
    ) -> PredictionResult:
        df = self._zone_to_df(zone)
        X_pre = self._preprocess(df)
        prob = float(self.model.predict_proba(X_pre)[0, 1])
        shap_explanations = explain_batch_from_df(
            self.model,
            self.preprocessor,
            self.feature_engineer,
            df,
            top_n=top_n,
        )
        return PredictionResult(
            zone_id=zone_id,
            suitability_score=round(prob, 4),
            is_recommended=prob >= self.threshold,
            top_features=shap_explanations[0],
        )

    def rank_zones(
        self,
        zones: list[ZoneInput | dict],
        zone_ids: list[str | int] | None = None,
        top_n_zones: int = 10,
    ) -> list[PredictionResult]:
        results = self.predict(zones, zone_ids)
        return sorted(results, key=lambda r: r.suitability_score, reverse=True)[:top_n_zones]
