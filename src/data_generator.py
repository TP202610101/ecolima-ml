"""
data_generator.py
Genera un dataset sintético que replica el perfil de features del schema real.
Sirve para desarrollar y testear el pipeline ANTES de tener datos INEI/OSM consolidados.

Distribuciones calibradas contra:
  - Datos reales de Miraflores y Magdalena del Mar (recycling_points)
  - Rangos típicos de densidad poblacional de Lima Metropolitana (INEI 2017)
  - Rangos de NSE de Lima Metropolitana (APEIM 2023)

Cuando los datos reales estén disponibles, este módulo se reemplaza por
los loaders de PostgreSQL/PostGIS (ver pipeline de ingesta en f4).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    SYNTH_DIR, ALL_FEATURES, TARGET_COLUMN, GROUP_COLUMN,
    DEMOGRAPHIC_FEATURES, GEOSPATIAL_FEATURES,
    OPERATIONAL_FEATURES, ENGINEERED_FEATURES,
    RANDOM_STATE,
)

# Distritos de Lima Metropolitana — subconjunto representativo
LIMA_DISTRICTS = {
    1:  "Miraflores",
    2:  "Magdalena del Mar",
    3:  "San Isidro",
    4:  "Barranco",
    5:  "Lince",
    6:  "Pueblo Libre",
    7:  "Jesús María",
    8:  "San Borja",
    9:  "Surquillo",
    10: "La Molina",
}

# Parámetros por distrito: (pop_density_mean, nse_ab_pct_mean, recycling_density_mean)
# Basados en INEI 2017 y APEIM 2023
DISTRICT_PROFILES = {
    1:  (17_000, 0.70, 2.5),   # Miraflores: alta densidad, NSE alto, buena cobertura
    2:  (14_000, 0.55, 1.5),   # Magdalena
    3:  ( 8_000, 0.80, 1.8),   # San Isidro
    4:  (12_000, 0.60, 0.8),   # Barranco
    5:  (20_000, 0.40, 0.6),   # Lince
    6:  (16_000, 0.45, 0.7),   # Pueblo Libre
    7:  (18_000, 0.50, 0.9),   # Jesús María
    8:  (10_000, 0.75, 1.2),   # San Borja
    9:  (22_000, 0.30, 0.4),   # Surquillo: alta densidad, NSE bajo, poca cobertura
    10: ( 5_000, 0.65, 0.5),   # La Molina: baja densidad, NSE alto
}

LAND_USE_CODES = {
    "residential": 0,
    "commercial":  1,
    "park":        2,
    "mixed":       3,
}


def _generate_district_batch(
    district_id: int,
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Genera n_samples zonas candidatas para un distrito dado."""
    pop_density_mean, nse_ab_mean, recycling_density_mean = DISTRICT_PROFILES[district_id]

    # ── Demográficas ──────────────────────────────────────────────────────────
    population_density = rng.normal(pop_density_mean, pop_density_mean * 0.15, n_samples).clip(500, 45_000)
    nse_ab_pct  = rng.normal(nse_ab_mean, 0.10, n_samples).clip(0.05, 0.95)
    nse_c_pct   = rng.normal(0.35, 0.08, n_samples).clip(0.05, 0.70)
    # NSE D+E ocupa el resto; forzar suma ≤ 1
    nse_de_pct  = (1 - nse_ab_pct - nse_c_pct).clip(0.0, 0.80)
    num_households   = (population_density * rng.uniform(0.08, 0.12, n_samples)).astype(int)
    urbanization_rate = rng.uniform(0.75, 0.99, n_samples)

    # ── Geoespaciales ─────────────────────────────────────────────────────────
    dist_nearest_road_m    = rng.exponential(30, n_samples).clip(1, 300)
    walkability_score      = rng.beta(5, 2, n_samples)           # sesgado hacia valores altos en Lima central
    poi_commercial_500m    = rng.poisson(recycling_density_mean * 15, n_samples)
    poi_educational_500m   = rng.poisson(3, n_samples)
    poi_parks_500m         = rng.poisson(1, n_samples)
    land_use_encoded       = rng.choice([0, 1, 2, 3], n_samples, p=[0.50, 0.25, 0.15, 0.10])
    area_m2                = rng.lognormal(mean=5.5, sigma=0.8, size=n_samples).clip(10, 5_000)

    # ── Operativas ────────────────────────────────────────────────────────────
    dist_nearest_recycling_m = rng.exponential(400, n_samples).clip(10, 3_000)
    recycling_density_1km    = rng.poisson(recycling_density_mean, n_samples)
    waste_per_capita_kg      = rng.normal(0.65, 0.15, n_samples).clip(0.2, 1.5)

    # brecha = proporción de población sin punto en 500m
    has_point_500m   = dist_nearest_recycling_m < 500
    coverage_gap_index = np.where(has_point_500m, rng.uniform(0, 0.3, n_samples),
                                                   rng.uniform(0.5, 1.0, n_samples))

    # ── Features derivadas ────────────────────────────────────────────────────
    # Accesibilidad compuesta: ponderación de walkability y distancia a vía
    accessibility_composite = (
        0.6 * walkability_score
        + 0.4 * (1 - dist_nearest_road_m / 300).clip(0, 1)
    )
    nse_high_ratio    = nse_ab_pct
    recycling_deficit = (~has_point_500m).astype(int)

    # ── Target (etiqueta de zona óptima) ─────────────────────────────────────
    # Lógica heurística que simula la "calidad" de una zona:
    # Más probable que sea óptima si: alta densidad, alta accesibilidad,
    # alta brecha de cobertura (necesita un punto) y suficiente área.
    logit = (
          1.5 * (population_density / 25_000)
        + 1.0 * accessibility_composite
        + 0.8 * coverage_gap_index
        + 0.5 * nse_high_ratio
        - 0.6 * (recycling_density_1km / 5)    # penaliza zonas ya bien cubiertas
        + rng.normal(0, 0.3, n_samples)         # ruido
    )
    prob_optimal = 1 / (1 + np.exp(-logit + 1.5))   # sigmoid centrada
    is_optimal   = (rng.uniform(0, 1, n_samples) < prob_optimal).astype(int)

    # Score continuo de idoneidad (0–1), por si el asesor confirma regresión
    suitability_score = prob_optimal.round(4)

    df = pd.DataFrame({
        GROUP_COLUMN: district_id,
        "district_name": LIMA_DISTRICTS[district_id],
        # Demográficas
        "population_density":  population_density.round(1),
        "num_households":      num_households,
        "nse_ab_pct":          nse_ab_pct.round(4),
        "nse_c_pct":           nse_c_pct.round(4),
        "nse_de_pct":          nse_de_pct.round(4),
        "urbanization_rate":   urbanization_rate.round(4),
        # Geoespaciales
        "dist_nearest_road_m":   dist_nearest_road_m.round(1),
        "walkability_score":     walkability_score.round(4),
        "poi_commercial_500m":   poi_commercial_500m,
        "poi_educational_500m":  poi_educational_500m,
        "poi_parks_500m":        poi_parks_500m,
        "land_use_encoded":      land_use_encoded,
        "area_m2":               area_m2.round(1),
        # Operativas
        "dist_nearest_recycling_m": dist_nearest_recycling_m.round(1),
        "recycling_density_1km":    recycling_density_1km,
        "waste_per_capita_kg":      waste_per_capita_kg.round(3),
        "coverage_gap_index":       coverage_gap_index.round(4),
        # Derivadas
        "accessibility_composite": accessibility_composite.round(4),
        "nse_high_ratio":          nse_high_ratio.round(4),
        "recycling_deficit":       recycling_deficit,
        # Targets
        "is_optimal":       is_optimal,
        "suitability_score": suitability_score,
    })

    return df


def generate_synthetic_dataset(
    n_per_district: int = 200,
    output_path: Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Genera dataset sintético completo con zonas candidatas para todos los distritos.

    Args:
        n_per_district: Número de zonas candidatas por distrito.
        output_path: Ruta de salida del CSV. Por defecto SYNTH_DIR/synthetic_dataset.csv
        save: Si True, guarda el CSV en output_path.

    Returns:
        DataFrame con todas las zonas candidatas.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    batches = []
    for district_id in LIMA_DISTRICTS:
        batch = _generate_district_batch(district_id, n_per_district, rng)
        batches.append(batch)

    df = pd.concat(batches, ignore_index=True)
    df.index.name = "zone_id"
    df.reset_index(inplace=True)

    if save:
        path = output_path or SYNTH_DIR / "synthetic_dataset.csv"
        df.to_csv(path, index=False)
        print(f"[data_generator] Dataset guardado: {path}  ({len(df)} filas)")

    # Resumen rápido de balance de clases
    balance = df["is_optimal"].value_counts(normalize=True)
    print(f"[data_generator] Balance de clases — is_optimal=1: {balance.get(1, 0):.1%}  is_optimal=0: {balance.get(0, 0):.1%}")

    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(n_per_district=200)
    print(df.describe().T[["mean", "std", "min", "max"]])
