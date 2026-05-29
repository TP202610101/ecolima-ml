"""
trainer.py
Entrenamiento de LightGBM con:
  - Optuna para HPO (objetivo: AUC en clasificación / RMSE en regresión)
  - GroupKFold para spatial cross-validation (grupos = distritos)
  - Early stopping con LightGBM callbacks

Uso rápido:
    from trainer import train
    result = train(X_train, y_train, groups_train)
"""

import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path

from config import (
    MODELS_DIR,
    LGBM_BASE_PARAMS,
    CV_N_SPLITS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from preprocessing import build_preprocessor, FeatureEngineer

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Objetivo Optuna ──────────────────────────────────────────────────────────

def _build_objective(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    Cierra sobre los datos de entrenamiento para crear la función objetivo de Optuna.
    Usa GroupKFold para que cada fold respete los límites distritales.
    """
    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    fe  = FeatureEngineer()
    pre = build_preprocessor()

    def objective(trial: optuna.Trial) -> float:
        params = {
            **LGBM_BASE_PARAMS,
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        }

        model = lgb.LGBMClassifier(**params)   # cambiar a LGBMRegressor si el target es continuo
        aucs  = []

        for train_idx, val_idx in gkf.split(X, y, groups):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Aplicar feature engineering + preprocesamiento en cada fold
            X_tr_eng  = fe.fit_transform(X_tr)
            X_val_eng = fe.transform(X_val)

            X_tr_pre  = pre.fit_transform(X_tr_eng)
            X_val_pre = pre.transform(X_val_eng)

            model.fit(
                X_tr_pre, y_tr,
                eval_set=[(X_val_pre, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )

            preds = model.predict_proba(X_val_pre)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))

        return np.mean(aucs)

    return objective


# ── Entrenamiento principal ──────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    run_hpo: bool = True,
    save: bool = True,
    model_name: str = "lgbm_recycling",
) -> dict:
    """
    Entrena el modelo LightGBM con HPO opcional y spatial CV.

    Args:
        X_train, y_train: Datos de entrenamiento.
        groups_train: Serie con district_id para GroupKFold.
        run_hpo: Si True, ejecuta Optuna antes del entrenamiento final.
        save: Si True, serializa modelo + metadata en MODELS_DIR.
        model_name: Nombre base del archivo de salida.

    Returns:
        dict con modelo entrenado, parámetros óptimos y métricas de CV.
    """
    best_params = LGBM_BASE_PARAMS.copy()

    if run_hpo:
        print(f"[trainer] Iniciando HPO con Optuna ({OPTUNA_N_TRIALS} trials, timeout {OPTUNA_TIMEOUT}s)...")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE),
        )
        objective = _build_objective(X_train, y_train, groups_train)
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)

        best_params.update(study.best_params)
        print(f"[trainer] Mejor AUC en CV: {study.best_value:.4f}")
        print(f"[trainer] Mejores parámetros: {study.best_params}")

    # Entrenamiento final sobre todo el conjunto de training
    fe  = FeatureEngineer()
    pre = build_preprocessor()

    X_eng = fe.fit_transform(X_train)
    X_pre = pre.fit_transform(X_eng)

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_pre, y_train)

    # CV final para reporte (sin HPO, solo para registrar la métrica)
    gkf  = GroupKFold(n_splits=CV_N_SPLITS)
    aucs = []
    for train_idx, val_idx in gkf.split(X_train, y_train, groups_train):
        X_tr,  X_val  = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr,  y_val  = y_train.iloc[train_idx], y_train.iloc[val_idx]
        X_tr_e  = fe.fit_transform(X_tr)
        X_val_e = fe.transform(X_val)
        X_tr_p  = pre.fit_transform(X_tr_e)
        X_val_p = pre.transform(X_val_e)
        tmp_model = lgb.LGBMClassifier(**best_params)
        tmp_model.fit(X_tr_p, y_tr, callbacks=[lgb.log_evaluation(-1)])
        aucs.append(roc_auc_score(y_val, tmp_model.predict_proba(X_val_p)[:, 1]))

    cv_auc_mean = float(np.mean(aucs))
    cv_auc_std  = float(np.std(aucs))
    print(f"[trainer] AUC CV final: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

    result = {
        "model":       final_model,
        "preprocessor": pre,
        "feature_engineer": fe,
        "best_params": best_params,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std":  cv_auc_std,
        "feature_names": ALL_FEATURES,
    }

    if save:
        _save_artifacts(result, model_name)

    return result


def _save_artifacts(result: dict, model_name: str) -> None:
    """Serializa modelo, preprocessor y metadata."""
    model_path = MODELS_DIR / f"{model_name}.joblib"
    meta_path  = MODELS_DIR / f"{model_name}_metadata.json"

    joblib.dump({
        "model":            result["model"],
        "preprocessor":     result["preprocessor"],
        "feature_engineer": result["feature_engineer"],
        "feature_names":    result["feature_names"],
    }, model_path)

    metadata = {
        "model_name":    model_name,
        "target":        TARGET_COLUMN,
        "cv_auc_mean":   result["cv_auc_mean"],
        "cv_auc_std":    result["cv_auc_std"],
        "best_params":   result["best_params"],
        "feature_names": result["feature_names"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[trainer] Modelo guardado: {model_path}")
    print(f"[trainer] Metadata guardada: {meta_path}")
