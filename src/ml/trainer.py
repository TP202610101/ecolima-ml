"""
trainer.py
Entrenamiento de LightGBM con:
  - Optuna para HPO (objetivo: AUC en clasificación)
  - GroupKFold para spatial cross-validation (grupos = distritos)
  - Early stopping con LightGBM callbacks

CAMBIOS v2 (2026-05-29):
  - Corregido UserWarning "X does not have valid feature names":
    se pasan feature_names explícitamente al fit() de LGBMClassifier
    después del ColumnTransformer para preservar los nombres de columna.
  - Silenciado optuna logging de forma más robusta.
"""

import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

from ml.config import (
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
from ml.preprocessing import build_preprocessor, FeatureEngineer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_feature_names_after_transform(preprocessor, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    """
    Devuelve los nombres de features en el orden que produce el ColumnTransformer.
    Necesario para evitar el UserWarning de LightGBM sobre feature names.
    """
    return numeric_features + categorical_features


def _numeric_features() -> list[str]:
    return [f for f in ALL_FEATURES if f not in CATEGORICAL_FEATURES]


def _make_cv(groups: pd.Series) -> tuple:
    """Devuelve (splitter, use_groups). Cae a StratifiedKFold si hay pocos distritos."""
    n_groups = groups.nunique()
    if n_groups >= CV_N_SPLITS:
        return GroupKFold(n_splits=CV_N_SPLITS), True
    n_splits = max(2, n_groups)
    print(f"[trainer] Solo {n_groups} grupo(s) en train → StratifiedKFold(n_splits={n_splits})")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), False


# ── Objetivo Optuna ──────────────────────────────────────────────────────────

def _build_objective(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    cv, use_groups = _make_cv(groups)
    fe             = FeatureEngineer()
    feature_names  = _get_feature_names_after_transform(None, _numeric_features(), CATEGORICAL_FEATURES)

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

        model = lgb.LGBMClassifier(**params)
        aucs  = []

        split_iter = cv.split(X, y, groups) if use_groups else cv.split(X, y)
        for train_idx, val_idx in split_iter:
            X_tr,  X_val  = X.iloc[train_idx], X.iloc[val_idx]
            y_tr,  y_val  = y.iloc[train_idx], y.iloc[val_idx]

            pre = build_preprocessor()

            X_tr_eng  = fe.fit_transform(X_tr)
            X_val_eng = fe.transform(X_val)

            X_tr_pre  = pd.DataFrame(pre.fit_transform(X_tr_eng),  columns=feature_names)
            X_val_pre = pd.DataFrame(pre.transform(X_val_eng),     columns=feature_names)

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
    best_params   = LGBM_BASE_PARAMS.copy()
    feature_names = _get_feature_names_after_transform(None, _numeric_features(), CATEGORICAL_FEATURES)

    if run_hpo:
        print(f"[trainer] Iniciando HPO con Optuna ({OPTUNA_N_TRIALS} trials, timeout {OPTUNA_TIMEOUT}s)...")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE),
        )
        objective = _build_objective(X_train, y_train, groups_train)
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)

        best_params.update(study.best_params)
        print(f"[trainer] Mejor AUC en CV (HPO): {study.best_value:.4f}")
        print(f"[trainer] Mejores parámetros: {study.best_params}")

    # Entrenamiento final
    fe  = FeatureEngineer()
    pre = build_preprocessor()

    X_eng = fe.fit_transform(X_train)
    X_pre = pd.DataFrame(pre.fit_transform(X_eng), columns=feature_names)

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_pre, y_train)

    # CV final para reporte de métrica
    cv_final, use_groups_final = _make_cv(groups_train)
    aucs = []
    split_iter = cv_final.split(X_train, y_train, groups_train) if use_groups_final else cv_final.split(X_train, y_train)
    for train_idx, val_idx in split_iter:
        X_tr,  X_val  = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr,  y_val  = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pre_fold = build_preprocessor()
        X_tr_e   = fe.fit_transform(X_tr)
        X_val_e  = fe.transform(X_val)
        X_tr_p   = pd.DataFrame(pre_fold.fit_transform(X_tr_e),  columns=feature_names)
        X_val_p  = pd.DataFrame(pre_fold.transform(X_val_e),     columns=feature_names)

        tmp = lgb.LGBMClassifier(**best_params)
        tmp.fit(X_tr_p, y_tr, callbacks=[lgb.log_evaluation(-1)])
        aucs.append(roc_auc_score(y_val, tmp.predict_proba(X_val_p)[:, 1]))

    cv_auc_mean = float(np.mean(aucs))
    cv_auc_std  = float(np.std(aucs))
    print(f"[trainer] AUC CV final: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

    result = {
        "model":            final_model,
        "preprocessor":     pre,
        "feature_engineer": fe,
        "best_params":      best_params,
        "cv_auc_mean":      cv_auc_mean,
        "cv_auc_std":       cv_auc_std,
        "feature_names":    feature_names,
    }

    if save:
        _save_artifacts(result, model_name)

    return result


def _save_artifacts(result: dict, model_name: str) -> None:
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
