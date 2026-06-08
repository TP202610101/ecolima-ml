"""
preprocessing.py
Pipeline de preprocesamiento con sklearn Pipeline + ColumnTransformer.
Diseñado para ser agnóstico a la fuente (datos sintéticos o reales INEI/OSM).

Decisiones de diseño:
  - Imputación por mediana (robusta a outliers) para todas las numéricas.
  - OrdinalEncoder para land_use_encoded (LightGBM lo maneja nativo como int).
  - No se aplica StandardScaler: LightGBM es invariante a escala.
  - El pipeline sklearn se puede serializar junto con el modelo (joblib).
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from ml.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DEMOGRAPHIC_FEATURES,
    GEOSPATIAL_FEATURES,
    OPERATIONAL_FEATURES,
    ENGINEERED_FEATURES,
    TARGET_COLUMN,
    GROUP_COLUMN,
)


# ── Transformer personalizado: recalcula features derivadas ──────────────────

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Recalcula/refresca las features derivadas a partir de las features base.
    Se ejecuta ANTES del ColumnTransformer para asegurarse de que las
    features derivadas estén siempre consistentes con los datos de entrada.

    Si los datos ya vienen con estas columnas calculadas (ej. del generador
    sintético o de PostGIS), el transformer las sobreescribe para garantizar
    consistencia.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Accesibilidad compuesta: walkability 60% + proximidad a vía 40%
        if "walkability_score" in X.columns and "dist_nearest_road_m" in X.columns:
            X["accessibility_composite"] = (
                0.6 * X["walkability_score"]
                + 0.4 * (1 - (X["dist_nearest_road_m"] / 300).clip(0, 1))
            )

        # Ratio NSE alto
        if "nse_ab_pct" in X.columns:
            X["nse_high_ratio"] = X["nse_ab_pct"]

        # Déficit de reciclaje: 1 si no hay punto en radio 500m
        if "dist_nearest_recycling_m" in X.columns:
            X["recycling_deficit"] = (X["dist_nearest_recycling_m"] >= 500).astype(int)

        # Índice de brecha de cobertura (si no viene calculado desde PostGIS)
        if "coverage_gap_index" not in X.columns:
            X["coverage_gap_index"] = X["recycling_deficit"].astype(float)

        # Compatibilidad con dataset real: has_park_300m → poi_parks_500m
        if "poi_parks_500m" not in X.columns and "has_park_300m" in X.columns:
            X["poi_parks_500m"] = X["has_park_300m"].astype(float)

        return X


# ── Pipelines por tipo de feature ────────────────────────────────────────────

def _numeric_features() -> list[str]:
    return [f for f in ALL_FEATURES if f not in CATEGORICAL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el ColumnTransformer que se integra en el Pipeline principal.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, _numeric_features()),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",           # descarta columnas no listadas (district_id, etc.)
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_full_pipeline(model) -> Pipeline:
    """
    Ensambla el pipeline completo:
      FeatureEngineer → ColumnTransformer → LightGBM model

    Args:
        model: Instancia de LGBMClassifier o LGBMRegressor ya configurada.

    Returns:
        sklearn Pipeline listo para fit/predict.
    """
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def load_and_split(
    csv_path: str | None = None,
    df: pd.DataFrame | None = None,
    test_districts: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Carga el dataset y realiza el split espacial (por distrito).

    El split espacial es fundamental para evitar data leakage:
    los distritos de test nunca aparecen en el set de entrenamiento.
    (Koldasbayeva et al., 2024)

    Args:
        csv_path: Ruta al CSV del dataset (sintético o real).
        df: DataFrame ya cargado (si se prefiere pasar directamente).
        test_districts: Lista de district_ids para test.
                        Si None, se usan los 2 distritos con mayor representación.

    Returns:
        X_train, X_test, y_train, y_test, groups_train, groups_test
    """
    if df is None and csv_path:
        df = pd.read_csv(csv_path)
    elif df is None:
        raise ValueError("Proveer csv_path o df.")

    # Seleccionar distritos de test (1 ó 2, garantizando que train no quede vacío)
    if test_districts is None:
        all_districts = sorted(df[GROUP_COLUMN].unique())
        n_test = min(2, len(all_districts) - 1)
        test_districts = all_districts[-n_test:]

    print(f"[preprocessing] Distritos de TEST: {test_districts}")
    print(f"[preprocessing] Distritos de TRAIN: {[d for d in df[GROUP_COLUMN].unique() if d not in test_districts]}")

    test_mask  = df[GROUP_COLUMN].isin(test_districts)
    train_mask = ~test_mask

    # Solo incluir features que existen en el CSV (engineered se calculan después)
    feature_cols = [f for f in ALL_FEATURES if f in df.columns]
    # Columnas base para FeatureEngineer aunque no estén en ALL_FEATURES
    extra_cols = [c for c in [
        "walkability_score", "dist_nearest_road_m", "nse_ab_pct",
        "dist_nearest_recycling_m", "coverage_gap_index", "has_park_300m",
    ] if c in df.columns]

    cols_to_keep = list(dict.fromkeys(feature_cols + extra_cols))

    X_train = df.loc[train_mask, cols_to_keep].copy()
    X_test  = df.loc[test_mask,  cols_to_keep].copy()
    y_train = df.loc[train_mask, TARGET_COLUMN]
    y_test  = df.loc[test_mask,  TARGET_COLUMN]
    groups_train = df.loc[train_mask, GROUP_COLUMN]
    groups_test  = df.loc[test_mask,  GROUP_COLUMN]

    print(f"[preprocessing] Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")
    print(f"[preprocessing] Balance train — is_optimal=1: {y_train.mean():.1%}")
    print(f"[preprocessing] Balance test  — is_optimal=1: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test, groups_train, groups_test
