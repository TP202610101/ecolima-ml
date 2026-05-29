"""
evaluator.py
Evaluación del modelo sobre el conjunto de test.

CAMBIOS v2 (2026-05-29):
  - find_optimal_threshold: rango de búsqueda cambiado de [0.1, 0.9]
    a [0.25, 0.75] para evitar umbrales degenerados que dan F1 alto
    trivialmente por predecir todo como clase mayoritaria.
  - Agregada advertencia cuando el umbral óptimo queda fuera de [0.35, 0.65].
  - evaluate_classifier: ahora recibe feature_names para pasar DataFrames
    al modelo y evitar warnings de sklearn/LightGBM.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from ml.config import TARGET_COLUMN


def evaluate_classifier(
    model,
    preprocessor,
    feature_engineer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    feature_names: list[str] | None = None,
    print_report: bool = True,
) -> dict:
    """
    Evalúa el modelo de clasificación sobre el conjunto de test.

    Args:
        threshold: Umbral de probabilidad para clasificar como óptima.
        feature_names: Lista de nombres de features post-transform.
                       Si se provee, los arrays se convierten a DataFrame
                       para evitar UserWarnings de sklearn/LightGBM.
    """
    X_eng  = feature_engineer.transform(X_test)
    X_pre_arr = preprocessor.transform(X_eng)

    if feature_names is not None:
        X_pre = pd.DataFrame(X_pre_arr, columns=feature_names)
    else:
        X_pre = X_pre_arr

    probs = model.predict_proba(X_pre)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "auc_roc":            float(roc_auc_score(y_test, probs)),
        "f1":                 float(f1_score(y_test, preds, zero_division=0)),
        "precision":          float(precision_score(y_test, preds, zero_division=0)),
        "recall":             float(recall_score(y_test, preds, zero_division=0)),
        "threshold":          threshold,
        "n_test":             len(y_test),
        "positive_rate_test": float(y_test.mean()),
        "positive_rate_pred": float(preds.mean()),
        "confusion_matrix":   confusion_matrix(y_test, preds).tolist(),
    }

    if print_report:
        print("\n── Evaluación en Test ──────────────────────────────────────")
        print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
        print(f"  F1-Score  : {metrics['f1']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  Umbral    : {threshold:.2f}")
        print(f"  Tasa real positivos (test): {metrics['positive_rate_test']:.1%}")
        print(f"  Tasa pred positivos (test): {metrics['positive_rate_pred']:.1%}")
        print(f"\n{classification_report(y_test, preds, zero_division=0)}")
        cm = np.array(metrics["confusion_matrix"])
        print(f"  Confusion Matrix:\n    TN={cm[0,0]}  FP={cm[0,1]}\n    FN={cm[1,0]}  TP={cm[1,1]}")

        # Advertencia de predicción degenerada
        if metrics["positive_rate_pred"] >= 0.95:
            print("\n  ⚠  ADVERTENCIA: el modelo predice casi todo como clase 1.")
            print("     AUC y F1 son métricas no confiables en este caso.")
            print("     Revisar balance de clases y offset del generador.")
        elif metrics["positive_rate_pred"] <= 0.05:
            print("\n  ⚠  ADVERTENCIA: el modelo predice casi todo como clase 0.")

    return metrics


def find_optimal_threshold(
    model,
    preprocessor,
    feature_engineer,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f1",
    feature_names: list[str] | None = None,
) -> float:
    """
    Busca el umbral óptimo en rango [0.25, 0.75] para evitar umbrales degenerados.

    FIX v2: rango anterior [0.1, 0.9] permitía encontrar umbral=0.10 que
    trivialmente maximiza F1 prediciendo todo como clase 1 con datasets
    desbalanceados. El rango [0.25, 0.75] fuerza al modelo a demostrar
    capacidad discriminativa real.
    """
    X_eng     = feature_engineer.transform(X_val)
    X_pre_arr = preprocessor.transform(X_eng)

    if feature_names is not None:
        X_pre = pd.DataFrame(X_pre_arr, columns=feature_names)
    else:
        X_pre = X_pre_arr

    probs = model.predict_proba(X_pre)[:, 1]

    # Rango conservador: evita umbrales degenerados
    thresholds = np.linspace(0.25, 0.75, 51)
    scores = []

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        if metric == "f1":
            scores.append(f1_score(y_val, preds, zero_division=0))
        elif metric == "precision":
            scores.append(precision_score(y_val, preds, zero_division=0))
        elif metric == "recall":
            scores.append(recall_score(y_val, preds, zero_division=0))

    best_idx = int(np.argmax(scores))
    best_thr = float(thresholds[best_idx])

    print(f"[evaluator] Umbral óptimo ({metric}): {best_thr:.2f}  (score={scores[best_idx]:.4f})")

    if best_thr <= 0.27 or best_thr >= 0.73:
        print(f"[evaluator] ⚠  Umbral en el límite del rango — señal del modelo débil")

    return best_thr
