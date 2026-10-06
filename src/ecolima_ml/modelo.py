# Fecha de creación: 2026-10-03
"""Regresión logística de propensión (propension-1.0.0) desde sus parámetros portables.

Usa `modelo/logistica_parametros.json` del paquete, sin pickle ni scikit-learn:

    z_num = (x' − media) / escala        con x' = log1p(max(x, 0)) en features_log1p, x' = x en el resto
    z_cat = indicadoras one-hot de clase_oportunidad (categoría desconocida: todas en 0)
    logit = intercepto + Σ coef · z
    propension = 1 / (1 + exp(−logit))

Contribución de un grupo de variables = Σ coef_j · (z_j − media de z_j en el entrenamiento), en logit;
cumple logit = logit_base + Σ contribuciones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ecolima_ml.config import AVISO_PROPENSION, VERSIONES
from ecolima_ml.paquete import Paquete, cargar


class ErrorFeatures(ValueError):
    """La observación no trae las features del modelo o sus valores no son válidos."""


@dataclass
class ResultadoPropension:
    propension: float
    logit: float
    contribuciones: dict[str, float]
    modelo: str
    aviso: str = field(default=AVISO_PROPENSION)


class ModeloPropension:
    def __init__(self, parametros: Mapping[str, Any], version: str = VERSIONES["modelo"]):
        p = parametros
        self.version = version
        self.features: list[str] = list(p["features"])
        self.numericas: list[str] = list(p["numericas"]["orden"])
        self.log1p = set(p["features_log1p"])
        self.media = np.asarray(p["numericas"]["media"], dtype=float)
        self.escala = np.asarray(p["numericas"]["escala"], dtype=float)
        self.categorica: str = p["categorica"]["feature"]
        self.categorias: list[str] = list(p["categorica"]["categorias"])
        self.columnas: list[str] = list(p["columnas_transformadas"])
        self.coef = np.asarray(p["coeficientes"], dtype=float)
        self.intercepto = float(p["intercepto"])
        c = p["contribuciones"]
        self.logit_base = float(c["logit_base"])
        self.medias_z = np.asarray(c["medias_z"], dtype=float)
        self.grupos: dict[str, list[str]] = {g: list(fs) for g, fs in c["grupos"].items()}
        self._comprobar_parametros()

    @classmethod
    def desde_paquete(cls, paquete: Paquete | None = None) -> "ModeloPropension":
        paquete = paquete or cargar()
        return cls(paquete.json("modelo/logistica_parametros.json"), version=paquete.versiones["modelo"])

    def _comprobar_parametros(self) -> None:
        esperadas = [f"num__{f}" for f in self.numericas] + [f"cat__{self.categorica}_{c}" for c in self.categorias]
        n = len(esperadas)
        if self.columnas != esperadas:
            raise ValueError("columnas_transformadas no corresponde a numericas + categorias")
        if not (len(self.coef) == len(self.medias_z) == n and len(self.media) == len(self.escala) == len(self.numericas)):
            raise ValueError("dimensiones inconsistentes en los parámetros del modelo")
        if set(self.features) != set(self.numericas) | {self.categorica}:
            raise ValueError("features no coincide con numericas + categorica")
        agrupadas = [f for fs in self.grupos.values() for f in fs]
        if sorted(agrupadas) != sorted(self.features):
            raise ValueError("los grupos de contribución no cubren exactamente las features")

    # ------------------------------------------------------------------ preparación
    def preparar(self, observaciones: pd.DataFrame | Mapping[str, Any] | list[Mapping[str, Any]],
                 permitir_desconocida: bool = False) -> np.ndarray:
        """Valida las features y devuelve la matriz z (n × columnas_transformadas).

        Una clase_oportunidad fuera de la enumeración se rechaza; con permitir_desconocida=True se
        aplica la regla de los parámetros (todas las indicadoras en 0).
        """
        df = pd.DataFrame([observaciones]) if isinstance(observaciones, Mapping) else pd.DataFrame(observaciones)
        faltan = [f for f in self.features if f not in df.columns]
        if faltan:
            raise ErrorFeatures(f"faltan features: {faltan}")
        try:
            X = df[self.numericas].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float, copy=True)
        except (ValueError, TypeError) as ex:
            raise ErrorFeatures(f"features numéricas con valores no numéricos: {ex}") from ex
        if not np.isfinite(X).all():
            raise ErrorFeatures("features numéricas con nulos o valores no finitos")
        clase = df[self.categorica]
        if clase.isna().any():
            raise ErrorFeatures(f"{self.categorica} nula")
        desconocidas = sorted(set(clase.astype(str)) - set(self.categorias))
        if desconocidas and not permitir_desconocida:
            raise ErrorFeatures(f"{self.categorica} fuera de {self.categorias}: {desconocidas}")
        log = [i for i, f in enumerate(self.numericas) if f in self.log1p]
        X[:, log] = np.log1p(np.clip(X[:, log], 0, None))
        zn = (X - self.media) / self.escala
        zc = np.column_stack([(clase == c).to_numpy(dtype=float) for c in self.categorias])
        return np.hstack([zn, zc])

    # ------------------------------------------------------------------ cálculo
    def logit(self, observaciones, permitir_desconocida: bool = False) -> np.ndarray:
        return self.intercepto + self.preparar(observaciones, permitir_desconocida) @ self.coef

    def propension(self, observaciones, permitir_desconocida: bool = False) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(observaciones, permitir_desconocida)))

    def contribuciones(self, observaciones, permitir_desconocida: bool = False) -> pd.DataFrame:
        """Aporte al logit de cada grupo de variables respecto al sitio medio de entrenamiento."""
        aporte = (self.preparar(observaciones, permitir_desconocida) - self.medias_z) * self.coef
        idx = {f"num__{f}": i for i, f in enumerate(self.numericas)}
        out = {}
        for g, fs in self.grupos.items():
            cols = []
            for f in fs:
                if f == self.categorica:
                    cols += [i for i, c in enumerate(self.columnas) if c.startswith(f"cat__{self.categorica}_")]
                else:
                    cols.append(idx[f"num__{f}"])
            out[g] = aporte[:, cols].sum(axis=1)
        return pd.DataFrame(out)

    def calcular(self, observacion: Mapping[str, Any], permitir_desconocida: bool = False) -> ResultadoPropension:
        """Propensión de una observación, con contribuciones por grupo y su interpretación."""
        z = self.logit(observacion, permitir_desconocida)[0]
        contrib = self.contribuciones(observacion, permitir_desconocida).iloc[0].to_dict()
        return ResultadoPropension(propension=1.0 / (1.0 + math.exp(-z)), logit=float(z),
                                   contribuciones={k: float(v) for k, v in contrib.items()}, modelo=self.version)
