# Fecha de creación: 2026-10-03
"""Reproducción de la regresión logística (contrato del paquete v1, §6, validación 16).

Reentrena con el dataset congelado, las 15 features oficiales, StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=semilla) con grupos = ubigeo y semillas 0–9, y compara con el paquete:

- métricas por semilla frente a modelo/metricas_por_semilla.csv;
- propensión fuera de fold por semilla frente a salidas/propension_oof_por_semilla.csv;
- modelo final (2 242 sitios) frente a modelo/logistica_parametros.json.

Con las versiones de librerías de manifest.json → entorno la coincidencia es exacta; con otras
versiones pueden aparecer diferencias numéricas pequeñas, que se informan con su magnitud.
Requiere scikit-learn y scipy (no los necesita el cálculo de la propensión).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ecolima_ml.paquete import Paquete

N_FOLDS = 5
TOPK = (0.05, 0.10, 0.20)
METRICAS = ["roc_auc", "pr_auc", "boyce", "prevalencia", "recall_top5", "recall_top10", "recall_top20"]


# ----------------------------------------------------------------------------- métricas
def boyce(score: np.ndarray, s: np.ndarray, ventanas: int = 100, ancho: float = 0.1) -> float:
    """Índice de Boyce continuo (Hirzel et al. 2006) sobre el percentil de la puntuación."""
    from scipy.stats import rankdata, spearmanr

    p = (rankdata(score) - 1) / (len(score) - 1)
    obs = p[s == 1]
    centros, pe = [], []
    for lo in np.linspace(0, 1 - ancho, ventanas):
        hi = lo + ancho
        esperado = np.mean((p >= lo) & (p <= hi))
        if esperado == 0:
            continue
        centros.append(lo + ancho / 2)
        pe.append(np.mean((obs >= lo) & (obs <= hi)) / esperado)
    if np.std(pe) == 0:
        return float("nan")
    return float(spearmanr(centros, pe).statistic)


def metricas(score: np.ndarray, s: np.ndarray) -> dict[str, float]:
    """ROC-AUC, PR-AUC, Boyce, prevalencia y recall en el 5, 10 y 20 % superior."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    orden = np.argsort(-score, kind="mergesort")
    m = {"roc_auc": roc_auc_score(s, score), "pr_auc": average_precision_score(s, score),
         "boyce": boyce(score, s), "prevalencia": float(s.mean())}
    for k in TOPK:
        top = orden[: int(np.ceil(k * len(score)))]
        m[f"recall_top{int(k * 100)}"] = float(s[top].sum() / s.sum())
    return m


# ----------------------------------------------------------------------------- modelo
def _preparar(df: pd.DataFrame, features: list[str], log1p: set[str]) -> pd.DataFrame:
    X = df[features].copy()
    for f in features:
        if f in log1p:
            X[f] = np.log1p(X[f].clip(lower=0))
    return X


def construir(features: list[str], categorica: str, categorias: list[str]):
    """Pipeline de la logística: estandarización + one-hot + LogisticRegression balanceada."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = [f for f in features if f != categorica]
    ct = ColumnTransformer([("num", StandardScaler(), num),
                            ("cat", OneHotEncoder(categories=[categorias], handle_unknown="ignore"), [categorica])])
    return make_pipeline(ct, LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0))


@dataclass
class Reproduccion:
    metricas: pd.DataFrame                       # calculadas, una fila por semilla
    referencia: pd.DataFrame                     # metricas_por_semilla.csv
    dif_metricas: dict[str, float]               # máx |calculada − referencia| por métrica
    dif_oof: float                               # máx |oof − propension_oof_por_semilla|
    dif_coeficientes: float                      # máx |coef final − logistica_parametros|
    folds: list[dict] = field(default_factory=list)

    def ok(self, tol: float = 1e-9) -> bool:
        return max([*self.dif_metricas.values(), self.dif_oof, self.dif_coeficientes]) <= tol

    @property
    def identica(self) -> bool:
        return self.ok(0.0)


def reproducir(paquete: Paquete, semillas: list[int] | None = None) -> Reproduccion:
    from sklearn.model_selection import StratifiedGroupKFold

    params = paquete.json("modelo/logistica_parametros.json")
    feats, log1p = list(params["features"]), set(params["features_log1p"])
    categorica = params["categorica"]["feature"]
    semillas = list(paquete.manifest["modelo"]["semillas"]) if semillas is None else semillas

    df = paquete.dataset()
    s = (df["rol"] == "positivo").to_numpy(int)
    categorias = sorted(df[categorica].unique())
    if categorias != params["categorica"]["categorias"]:
        raise ValueError(f"categorías del dataset {categorias} ≠ parámetros {params['categorica']['categorias']}")
    X = _preparar(df, feats, log1p)

    filas, folds, oofs = [], [], {}
    for sem in semillas:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=sem)
        oof = np.full(len(df), np.nan)
        for k, (itr, ite) in enumerate(cv.split(df, s, groups=df["ubigeo"])):
            pipe = construir(feats, categorica, categorias).fit(X.iloc[itr], s[itr])
            oof[ite] = pipe.predict_proba(X.iloc[ite])[:, 1]
            folds.append({"semilla": sem, "fold": k, "n_test": len(ite), "positivos_test": int(s[ite].sum()),
                          "distritos_test": int(df["ubigeo"].iloc[ite].nunique())})
        oofs[sem] = oof
        filas.append({"semilla": sem, **metricas(oof, s)})
    met = pd.DataFrame(filas)

    ref = paquete.csv("modelo/metricas_por_semilla.csv", float_precision="round_trip").set_index("semilla").loc[semillas]
    dif_met = {m: float(np.nanmax(np.abs(met.set_index("semilla")[m].to_numpy() - ref[m].to_numpy()))) for m in METRICAS}

    ref_oof = paquete.csv("salidas/propension_oof_por_semilla.csv", float_precision="round_trip").set_index("sitio_id")
    ref_oof = ref_oof.loc[df["sitio_id"]]
    dif_oof = max(float(np.abs(oofs[sem] - ref_oof[f"semilla_{sem}"].to_numpy()).max()) for sem in semillas)

    final = construir(feats, categorica, categorias).fit(X, s)
    lr = final[-1]
    dif_coef = float(max(np.abs(lr.coef_[0] - np.asarray(params["coeficientes"])).max(),
                         abs(lr.intercept_[0] - params["intercepto"])))
    return Reproduccion(met, ref.reset_index(), dif_met, dif_oof, dif_coef, folds)
