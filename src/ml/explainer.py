"""
explainer.py
SHAP TreeExplainer para LightGBM.
Genera explicaciones por instancia y extrae top-N features para la API.

El explainer opera sobre los datos YA preprocesados (output del ColumnTransformer),
con el mismo feature_names_out del preprocessor para mantener los nombres legibles.
"""

import shap
import numpy as np
import pandas as pd

from ml.config import ALL_FEATURES, SHAP_TOP_N_FEATURES


def build_explainer(model) -> shap.TreeExplainer:
    """
    Construye el TreeExplainer de SHAP para el modelo LightGBM.
    O(T·L) por instancia, sin necesidad de background dataset para TreeExplainer.
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def explain_instance(
    explainer: shap.TreeExplainer,
    X_preprocessed: np.ndarray,
    feature_names: list[str],
    top_n: int = SHAP_TOP_N_FEATURES,
) -> list[dict]:
    """
    Genera explicaciones SHAP para un batch de instancias preprocesadas.

    Args:
        X_preprocessed: Array 2D (n_instances, n_features) — output del preprocessor.
        feature_names: Nombres de features en el mismo orden que X_preprocessed.
        top_n: Número de features a retornar por instancia.

    Returns:
        Lista de dicts, uno por instancia:
        [
          {
            "feature": "coverage_gap_index",
            "shap_value": 0.34,
            "direction": "positive"    # "positive" = empuja hacia zona óptima
          },
          ...
        ]
    """
    shap_values = explainer.shap_values(X_preprocessed)

    # LightGBM binario retorna lista de 2 arrays (clase 0 y clase 1)
    # Tomamos los SHAP values de la clase positiva (índice 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    results = []
    for instance_shap in shap_values:
        # Ordenar por magnitud absoluta descendente
        sorted_idx  = np.argsort(np.abs(instance_shap))[::-1][:top_n]
        top_features = [
            {
                "feature":    feature_names[i],
                "shap_value": round(float(instance_shap[i]), 4),
                "direction":  "positive" if instance_shap[i] > 0 else "negative",
            }
            for i in sorted_idx
        ]
        results.append(top_features)

    return results


def explain_batch_from_df(
    model,
    preprocessor,
    feature_engineer,
    X: pd.DataFrame,
    top_n: int = SHAP_TOP_N_FEATURES,
) -> list[list[dict]]:
    """
    Wrapper de alto nivel: aplica FE + preprocesamiento + SHAP en un solo paso.
    Para usar directamente desde predictor.py o desde el endpoint FastAPI.

    Returns:
        Lista de listas de dicts con las top-N features SHAP por instancia.
    """
    X_eng = feature_engineer.transform(X)
    X_pre = preprocessor.transform(X_eng)

    feature_names = _get_feature_names(preprocessor)
    explainer     = build_explainer(model)

    return explain_instance(explainer, X_pre, feature_names, top_n)


def _get_feature_names(preprocessor) -> list[str]:
    """
    Extrae los nombres de features del ColumnTransformer en el mismo orden
    en que aparecen en el output transformado.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback: usar ALL_FEATURES (mismo orden que ColumnTransformer)
        return ALL_FEATURES
