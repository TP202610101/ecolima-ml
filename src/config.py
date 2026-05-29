"""
config.py
Constantes globales del módulo ML: rutas, listas de features, parámetros por defecto.
"""

from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SYNTH_DIR  = DATA_DIR / "synthetic"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

# ── Features ─────────────────────────────────────────────────────────────────
# Demográficas (fuente: INEI / socioeconomic_indicators)
DEMOGRAPHIC_FEATURES = [
    "population_density",       # hab/km²
    "num_households",           # número de hogares en zona censal
    "nse_ab_pct",               # % NSE A+B (proxy de participación en reciclaje)
    "nse_c_pct",
    "nse_de_pct",
    "urbanization_rate",        # tasa de urbanización (0–1)
]

# Geoespaciales (fuente: OSM / road_network / candidate_zones)
GEOSPATIAL_FEATURES = [
    "dist_nearest_road_m",      # distancia al nodo vial principal más cercano (m)
    "walkability_score",        # isócrona 5 min a pie (0–1)
    "poi_commercial_500m",      # densidad POIs comerciales en radio 500m
    "poi_educational_500m",     # densidad POIs educativos en radio 500m
    "poi_parks_500m",           # densidad parques en radio 500m
    "land_use_encoded",         # residencial=0, comercial=1, parque=2, mixto=3
    "area_m2",                  # área disponible del terreno candidato
]

# Operativas (fuente: recycling_points / waste_generation)
OPERATIONAL_FEATURES = [
    "dist_nearest_recycling_m", # distancia al punto de reciclaje más cercano (m)
    "recycling_density_1km",    # puntos de reciclaje existentes en radio 1km
    "waste_per_capita_kg",      # generación de residuos por habitante (kg/día)
    "coverage_gap_index",       # índice de brecha de cobertura (0–1)
]

# Derivadas (feature engineering)
ENGINEERED_FEATURES = [
    "accessibility_composite",  # score compuesto (ponderación de distancias)
    "nse_high_ratio",           # ratio NSE A+B / total
    "recycling_deficit",        # 1 si no hay punto en radio 500m, else 0
]

ALL_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + GEOSPATIAL_FEATURES
    + OPERATIONAL_FEATURES
    + ENGINEERED_FEATURES
)

# Features categóricas que LightGBM maneja de forma nativa
CATEGORICAL_FEATURES = ["land_use_encoded"]

# Columna target — cambiar según confirmación del asesor:
#   "is_optimal"  → clasificación binaria (0/1)
#   "suitability_score" → regresión (0.0–1.0)
TARGET_COLUMN = "is_optimal"        # ⚠ PENDIENTE confirmación asesor

# Columna de grupo para spatial CV (split por distrito)
GROUP_COLUMN = "district_id"

# ── Parámetros LightGBM base ─────────────────────────────────────────────────
LGBM_BASE_PARAMS = {
    "objective": "binary",          # cambiar a "regression" si el target es continuo
    "metric": "auc",                # cambiar a "rmse" para regresión
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,        # regularización clave para datasets pequeños
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ── Parámetros de validación ─────────────────────────────────────────────────
CV_N_SPLITS = 5
TRAIN_TEST_SPLIT = 0.70             # 70% train, 30% test (split espacial)
RANDOM_STATE = 42

# ── Optuna ───────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT  = 600               # segundos

# ── SHAP ─────────────────────────────────────────────────────────────────────
SHAP_TOP_N_FEATURES = 5             # top N features a exponer en la API
