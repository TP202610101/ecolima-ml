"""SHAP helpers for LightGBM explanations.

SHAP is optional so the prediction service can run basic scoring without the
heavy explanation dependency installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import ALL_FEATURES, SHAP_TOP_N_FEATURES

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover - exercised only when shap is absent
    shap = None


def build_explainer(model):
    """Build a TreeExplainer or raise a clear dependency error."""
    if shap is None:
        raise RuntimeError("SHAP is not installed. Install requirements-api.txt to enable explanations.")
    return shap.TreeExplainer(model)


def explain_instance(
    explainer,
    X_preprocessed: np.ndarray,
    feature_names: list[str],
    top_n: int = SHAP_TOP_N_FEATURES,
) -> list[dict]:
    """Generate top-N SHAP explanations for a preprocessed batch."""
    shap_values = explainer.shap_values(X_preprocessed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    results = []
    for instance_shap in shap_values:
        sorted_idx = np.argsort(np.abs(instance_shap))[::-1][:top_n]
        results.append([
            {
                "feature": feature_names[i],
                "shap_value": round(float(instance_shap[i]), 4),
                "direction": "positive" if instance_shap[i] > 0 else "negative",
            }
            for i in sorted_idx
        ])
    return results


def explain_batch_from_df(
    model,
    preprocessor,
    feature_engineer,
    X: pd.DataFrame,
    top_n: int = SHAP_TOP_N_FEATURES,
) -> list[list[dict]]:
    """Apply feature engineering, preprocessing and SHAP in one step."""
    X_eng = feature_engineer.transform(X)
    X_pre = preprocessor.transform(X_eng)
    feature_names = _get_feature_names(preprocessor)
    explainer = build_explainer(model)
    return explain_instance(explainer, X_pre, feature_names, top_n)


def _get_feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        return ALL_FEATURES
