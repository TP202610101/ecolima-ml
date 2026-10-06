# Fecha de creación: 2026-10-03
"""Validación 16: reentrenar la logística reproduce las métricas de referencia del paquete (unos segundos)."""

import platform
from importlib.metadata import version

import pytest

from ecolima_ml.paquete import cargar
from ecolima_ml.reproducir import METRICAS, reproducir


@pytest.fixture(scope="module")
def paquete(ruta_paquete):
    return cargar(ruta_paquete)


def _entorno_del_paquete(paquete) -> bool:
    e = paquete.manifest["entorno"]
    return (platform.python_version() == e["python"]
            and all(version(d) == e[k] for k, d in {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
                                                    "scikit-learn": "scikit-learn"}.items()))


def test_reproduce_metricas_por_semilla(paquete):
    r = reproducir(paquete)
    assert list(r.metricas["semilla"]) == list(range(10))
    assert len(r.folds) == 50
    assert all(f["positivos_test"] > 0 for f in r.folds)
    # Con el entorno del manifiesto la coincidencia es exacta; con otro, dentro de 1e-9.
    tol = 0.0 if _entorno_del_paquete(paquete) else 1e-9
    assert r.ok(tol), {**r.dif_metricas, "oof": r.dif_oof, "coef": r.dif_coeficientes}
    assert set(r.dif_metricas) == set(METRICAS)
