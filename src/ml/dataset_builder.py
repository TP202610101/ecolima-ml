"""
Utilities to build auditable real candidate datasets and simulated training data.

The real dataset prepared here is provisional: it keeps legacy labels as audit
metadata but does not promote them to the thesis target. The simulated dataset is
explicitly for development and endpoint testing while the real labeling contract
is still being closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.config import (
    ALL_FEATURES,
    DEFAULT_REAL_CANDIDATE_DATASET,
    DEFAULT_SIMULATED_TRAINING_DATASET,
    REAL_DIR,
    SYNTH_DIR,
)
from ml.data_generator import generate_synthetic_dataset

REAL_DATASET_VERSION = "real-candidates-v0.3"
SIMULATED_DATASET_VERSION = "simulated-training-v0.3"
LABEL_VERSION = "is-suitable-v0"

TRACE_COLUMNS = [
    "candidate_id",
    "dataset_version",
    "label_version",
    "label_status",
    "is_suitable",
    "candidate_origin",
    "source_dataset",
    "evidence_summary",
    "label_reason",
    "review_notes",
]

LOCATION_COLUMNS = [
    "district_id",
    "district",
    "latitude",
    "longitude",
    "centroid_lat",
    "centroid_lon",
    "geometry_quality",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _materialized_real_source() -> Path:
    materialized = REAL_DIR / "dataset_entrenamiento_v1_materialized.csv"
    raw = REAL_DIR / "dataset_entrenamiento_v1.csv"
    return materialized if materialized.exists() else raw


def _ensure_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "poi_parks_500m" not in df.columns and "has_park_300m" in df.columns:
        df["poi_parks_500m"] = df["has_park_300m"].astype(float)
    if "accessibility_composite" not in df.columns and {"walkability_score", "dist_nearest_road_m"}.issubset(df.columns):
        df["accessibility_composite"] = (
            0.6 * df["walkability_score"]
            + 0.4 * (1 - (df["dist_nearest_road_m"] / 300).clip(0, 1))
        )
    if "nse_high_ratio" not in df.columns and "nse_ab_pct" in df.columns:
        df["nse_high_ratio"] = df["nse_ab_pct"]
    if "recycling_deficit" not in df.columns and "dist_nearest_recycling_m" in df.columns:
        df["recycling_deficit"] = (df["dist_nearest_recycling_m"] >= 500).astype(int)
    if "coverage_gap_index" not in df.columns and "recycling_deficit" in df.columns:
        df["coverage_gap_index"] = df["recycling_deficit"].astype(float)
    return df


def prepare_real_candidate_dataset(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a provisional real candidate dataset from the legacy real CSV."""
    source = Path(input_path) if input_path else _materialized_real_source()
    output = Path(output_path) if output_path else DEFAULT_REAL_CANDIDATE_DATASET
    summary_output = Path(summary_path) if summary_path else output.with_suffix(".metadata.json")

    if not source.exists():
        raise FileNotFoundError(f"Real source dataset not found: {source}")

    df = pd.read_csv(source)
    df = _ensure_engineered_features(df)

    legacy_target = df["is_optimal"].copy() if "is_optimal" in df.columns else pd.Series([pd.NA] * len(df))
    if "is_optimal" in df.columns:
        df = df.drop(columns=["is_optimal"])

    if "centroid_lat" not in df.columns:
        df["centroid_lat"] = pd.NA
    if "centroid_lon" not in df.columns:
        df["centroid_lon"] = pd.NA
    if "latitude" in df.columns:
        df["centroid_lat"] = df["centroid_lat"].fillna(df["latitude"])
    if "longitude" in df.columns:
        df["centroid_lon"] = df["centroid_lon"].fillna(df["longitude"])

    district = df["Distrito"] if "Distrito" in df.columns else df.get("district_name", pd.Series([""] * len(df)))
    df.insert(0, "candidate_id", [f"ECO-REAL-V03-{i:04d}" for i in range(1, len(df) + 1)])
    df["dataset_version"] = REAL_DATASET_VERSION
    df["label_version"] = LABEL_VERSION
    df["label_status"] = "desconocido"
    df["is_suitable"] = ""
    df["is_optimal_legacy"] = legacy_target
    df["candidate_origin"] = np.where(
        legacy_target.fillna(0).astype(int) == 1,
        "legacy_positive_existing_point",
        "legacy_unobserved_cell_or_negative",
    )
    df["source_dataset"] = str(source.as_posix())
    df["district"] = district
    df["geometry_quality"] = np.where(
        df["centroid_lat"].notna() & df["centroid_lon"].notna(),
        "centroid_legacy_available",
        "centroid_missing_review_required",
    )
    df["evidence_summary"] = "Fila real heredada del dataset ML v1; conserva features como evidencia tecnica provisional."
    df["label_reason"] = "La etiqueta heredada no se adopta como is_suitable; requiere contrato probado y evidencia independiente."
    df["review_notes"] = "No usar como dataset supervisado final de tesis; sirve para endpoints y auditoria tecnica provisional."

    for column in ALL_FEATURES:
        if column not in df.columns:
            df[column] = pd.NA

    ordered = [c for c in TRACE_COLUMNS + LOCATION_COLUMNS + ALL_FEATURES if c in df.columns]
    remainder = [c for c in df.columns if c not in ordered]
    df = df[ordered + remainder]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    summary = {
        "dataset_version": REAL_DATASET_VERSION,
        "source": str(source),
        "output": str(output),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "label_status_counts": df["label_status"].value_counts(dropna=False).to_dict(),
        "legacy_target_counts": pd.Series(legacy_target).value_counts(dropna=False).to_dict(),
        "with_centroid": int((df["centroid_lat"].notna() & df["centroid_lon"].notna()).sum()),
        "training_allowed": False,
        "warning": "Dataset real provisional para auditoria/endpoints; no valida el target ni reemplaza etiquetas trazables.",
    }
    _write_summary(summary_output, summary)
    return df, summary


def build_simulated_training_dataset(
    n_per_district: int = 100,
    positive_rate: float = 0.35,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a simulated, trainable dataset with a controlled positive rate."""
    if not 0.05 <= positive_rate <= 0.80:
        raise ValueError("positive_rate must be between 0.05 and 0.80")

    output = Path(output_path) if output_path else DEFAULT_SIMULATED_TRAINING_DATASET
    summary_output = Path(summary_path) if summary_path else output.with_suffix(".metadata.json")

    df = generate_synthetic_dataset(n_per_district=n_per_district, save=False)
    cutoff = float(df["suitability_score"].quantile(1 - positive_rate))
    df["is_optimal"] = (df["suitability_score"] >= cutoff).astype(int)
    df["is_suitable"] = df["is_optimal"]

    df.insert(0, "candidate_id", [f"ECO-SIM-V03-{i:04d}" for i in range(1, len(df) + 1)])
    df["dataset_version"] = SIMULATED_DATASET_VERSION
    df["label_version"] = "simulated-is-suitable-v0.3"
    df["label_status"] = "etiquetado_simulado"
    df["candidate_origin"] = "simulated_candidate_cell"
    df["source_dataset"] = "ml.data_generator.generate_synthetic_dataset"
    df["evidence_summary"] = "Registro simulado para desarrollo tecnico de entrenamiento y endpoints."
    df["label_reason"] = "Etiqueta simulada derivada de suitability_score y tasa positiva controlada; no es evidencia real de tesis."
    df["review_notes"] = "Usar para desarrollo, pruebas y contratos de API; no reportar como resultado real."

    ordered = [
        "candidate_id", "dataset_version", "label_version", "label_status", "is_suitable",
        "candidate_origin", "source_dataset", "district_id", "district_name", "zone_id",
    ]
    ordered += [c for c in ALL_FEATURES if c in df.columns]
    ordered += ["is_optimal", "suitability_score", "evidence_summary", "label_reason", "review_notes"]
    remainder = [c for c in df.columns if c not in ordered]
    df = df[ordered + remainder]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    summary = {
        "dataset_version": SIMULATED_DATASET_VERSION,
        "output": str(output),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "districts": int(df["district_id"].nunique()),
        "n_per_district": int(n_per_district),
        "target_positive_rate_requested": float(positive_rate),
        "target_positive_rate_observed": float(df["is_optimal"].mean()),
        "target_counts": df["is_optimal"].value_counts(dropna=False).to_dict(),
        "score_cutoff": cutoff,
        "training_allowed": True,
        "warning": "Dataset simulado para desarrollo tecnico; no usar como evidencia empirica de tesis.",
    }
    _write_summary(summary_output, summary)
    return df, summary
