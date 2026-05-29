"""
evaluator.py
Evaluación del modelo sobre el conjunto de test.
Genera reporte de métricas estructurado (clasificación o regresión).
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from config import TARGET_COLUMN


def evaluate_classifier(
    model,
    preprocessor,
    feature_engineer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    print_report: bool = True,
) -> dict:
    """
    Evalúa el modelo de clasificación sobre el conjunto de test.

    Args:
        threshold: Umbral de probabilidad para clasificar como óptima.
                   El valor por defecto (0.5) puede ajustarse si hay
                   desbalance de clases o si el asesor prioriza recall.

    Returns:
        dict con AUC-ROC, F1, Precision, Recall y confusion matrix.
    """
    X_eng  = feature_engineer.transform(X_test)
    X_pre  = preprocessor.transform(X_eng)
    probs  = model.predict_proba(X_pre)[:, 1]
    preds  = (probs >= threshold).astype(int)

    metrics = {
        "auc_roc":   float(roc_auc_score(y_test, probs)),
        "f1":        float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall":    float(recall_score(y_test, preds, zero_division=0)),
        "threshold": threshold,
        "n_test":    len(y_test),
        "positive_rate_test": float(y_test.mean()),
        "positive_rate_pred": float(preds.mean()),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }

    if print_report:
        print("\n── Evaluación en Test ──────────────────────────────────────")
        print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
        print(f"  F1-Score  : {metrics['f1']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  Umbral    : {threshold}")
        print(f"\n{classification_report(y_test, preds, zero_division=0)}")
        cm = np.array(metrics["confusion_matrix"])
        print(f"  Confusion Matrix:\n    TN={cm[0,0]}  FP={cm[0,1]}\n    FN={cm[1,0]}  TP={cm[1,1]}")

    return metrics


def find_optimal_threshold(
    model,
    preprocessor,
    feature_engineer,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f1",
) -> float:
    """
    Busca el umbral óptimo que maximiza F1 (o precision/recall) en validación.
    Útil cuando hay desbalance de clases.

    Returns:
        Umbral óptimo (float entre 0 y 1).
    """
    X_eng = feature_engineer.transform(X_val)
    X_pre = preprocessor.transform(X_eng)
    probs = model.predict_proba(X_pre)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 81)
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
    print(f"[evaluator] Umbral óptimo para {metric}: {best_thr:.2f} (score={scores[best_idx]:.4f})")
    return best_thr
