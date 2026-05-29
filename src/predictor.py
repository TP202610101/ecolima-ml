"""
predictor.py
Interfaz de inferencia limpia para FastAPI.
Carga el modelo serializado y expone predict() y predict_with_explanation().

Este módulo es el único punto de contacto entre el módulo ML y el backend.
FastAPI importa únicamente este archivo.

Uso:
    from predictor import RecyclingPredictor
    predictor = RecyclingPredictor.load("lgbm_recycling")
    result = predictor.predict_with_explanation(zone_data)
"""

import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

from config import MODELS_DIR, SHAP_TOP_N_FEATURES, ALL_FEATURES
from explainer import explain_batch_from_df


# ── Schema de entrada (equivalente al Pydantic model en FastAPI) ─────────────

@dataclass
class ZoneInput:
    """
    Datos de una zona candidata. Equivalente al request body del endpoint POST /predict.
    FastAPI usará un Pydantic BaseModel análogo a esto.
    """
    population_density:        float
    num_households:            int
    nse_ab_pct:                float
    nse_c_pct:                 float
    nse_de_pct:                float
    urbanization_rate:         float
    dist_nearest_road_m:       float
    walkability_score:         float
    poi_commercial_500m:       int
    poi_educational_500m:      int
    poi_parks_500m:            int
    land_use_encoded:          int       # 0=residencial, 1=comercial, 2=parque, 3=mixto
    area_m2:                   float
    dist_nearest_recycling_m:  float
    recycling_density_1km:     int
    waste_per_capita_kg:       float
    coverage_gap_index:        float
    # Derivadas — se recalculan en FeatureEngineer si no se proveen
    accessibility_composite:   float | None = None
    nse_high_ratio:            float | None = None
    recycling_deficit:         int   | None = None


# ── Schema de salida ─────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    zone_id:          str | int
    suitability_score: float               # probabilidad de zona óptima (0–1)
    is_recommended:   bool                 # True si score >= threshold
    top_features:     list[dict] = field(default_factory=list)   # SHAP top-N


# ── Predictor ────────────────────────────────────────────────────────────────

class RecyclingPredictor:
    """
    Wrapper de inferencia para el modelo LightGBM serializado.
    Se instancia una sola vez al arrancar el servidor FastAPI (singleton).
    """

    def __init__(self, model, preprocessor, feature_engineer, feature_names, metadata):
        self.model            = model
        self.preprocessor     = preprocessor
        self.feature_engineer = feature_engineer
        self.feature_names    = feature_names
        self.metadata         = metadata
        self.threshold        = 0.5         # ajustar post-evaluación si hay desbalance

    @classmethod
    def load(cls, model_name: str = "lgbm_recycling") -> "RecyclingPredictor":
        """Carga el modelo serializado desde MODELS_DIR."""
        model_path = MODELS_DIR / f"{model_name}.joblib"
        meta_path  = MODELS_DIR / f"{model_name}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        artifacts = joblib.load(model_path)
        metadata  = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        return cls(
            model            = artifacts["model"],
            preprocessor     = artifacts["preprocessor"],
            feature_engineer = artifacts["feature_engineer"],
            feature_names    = artifacts.get("feature_names", ALL_FEATURES),
            metadata         = metadata,
        )

    def _zone_to_df(self, zone: ZoneInput | dict) -> pd.DataFrame:
        """Convierte una zona (dataclass o dict) a DataFrame de una fila."""
        if isinstance(zone, dict):
            data = zone
        else:
            data = {k: v for k, v in zone.__dict__.items() if v is not None}
        return pd.DataFrame([data])

    def predict(
        self,
        zones: list[ZoneInput | dict],
        zone_ids: list[str | int] | None = None,
    ) -> list[PredictionResult]:
        """
        Predicción batch sin explicaciones SHAP.
        Más rápido — usar para endpoints de listado masivo.
        """
        df = pd.concat([self._zone_to_df(z) for z in zones], ignore_index=True)

        X_eng = self.feature_engineer.transform(df)
        X_pre = self.preprocessor.transform(X_eng)
        probs = self.model.predict_proba(X_pre)[:, 1]

        ids = zone_ids or list(range(len(zones)))
        return [
            PredictionResult(
                zone_id           = ids[i],
                suitability_score = round(float(probs[i]), 4),
                is_recommended    = bool(probs[i] >= self.threshold),
            )
            for i in range(len(zones))
        ]

    def predict_with_explanation(
        self,
        zone: ZoneInput | dict,
        zone_id: str | int = 0,
        top_n: int = SHAP_TOP_N_FEATURES,
    ) -> PredictionResult:
        """
        Predicción individual con explicación SHAP.
        Para el endpoint POST /predict (analista municipal).
        """
        df = self._zone_to_df(zone)

        X_eng = self.feature_engineer.transform(df)
        X_pre = self.preprocessor.transform(X_eng)
        prob  = float(self.model.predict_proba(X_pre)[0, 1])

        shap_explanations = explain_batch_from_df(
            self.model, self.preprocessor, self.feature_engineer, df, top_n=top_n
        )

        return PredictionResult(
            zone_id           = zone_id,
            suitability_score = round(prob, 4),
            is_recommended    = prob >= self.threshold,
            top_features      = shap_explanations[0],
        )

    def rank_zones(
        self,
        zones: list[ZoneInput | dict],
        zone_ids: list[str | int] | None = None,
        top_n_zones: int = 10,
    ) -> list[PredictionResult]:
        """
        Predice y retorna las top_n_zones más recomendadas ordenadas por score.
        Para el endpoint GET /recommendations.
        """
        results = self.predict(zones, zone_ids)
        return sorted(results, key=lambda r: r.suitability_score, reverse=True)[:top_n_zones]
