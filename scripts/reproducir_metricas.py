#!/usr/bin/env python3
# Fecha de creación: 2026-10-03
"""Reentrena la regresión logística y compara con las métricas de referencia del paquete (validación 16).

    python scripts/reproducir_metricas.py [--paquete paquete/v1.0.0] [--tol 1e-9]

Sale con código 0 si métricas por semilla, propensión fuera de fold y coeficientes finales coinciden
dentro de la tolerancia; 1 en otro caso. Avisa si las versiones de las librerías difieren del manifiesto.
"""

from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecolima_ml.config import RUTA_PAQUETE  # noqa: E402
from ecolima_ml.paquete import cargar  # noqa: E402
from ecolima_ml.reproducir import METRICAS, reproducir  # noqa: E402

LIBRERIAS = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "scikit-learn"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paquete", type=Path, default=RUTA_PAQUETE)
    ap.add_argument("--tol", type=float, default=1e-9, help="diferencia absoluta máxima admitida")
    a = ap.parse_args()

    paq = cargar(a.paquete)
    entorno = paq.manifest["entorno"]
    distintas = [f"{k} {version(d)} (paquete: {entorno[k]})" for k, d in LIBRERIAS.items() if version(d) != entorno.get(k)]
    if platform.python_version() != entorno.get("python"):
        distintas.append(f"python {platform.python_version()} (paquete: {entorno['python']})")
    if distintas:
        print("AVISO: versiones distintas a las del manifiesto; pueden aparecer diferencias numéricas pequeñas:")
        for d in distintas:
            print(f"  - {d}")

    r = reproducir(paq)
    print(f"Modelo {paq.versiones['modelo']} · {len(r.metricas)} semillas × 5 folds agrupados por ubigeo")
    print(r.metricas.round(4).to_string(index=False))
    print("\nmáx |calculada − referencia|:")
    for m in METRICAS:
        print(f"  {m:<13} {r.dif_metricas[m]:.3g}")
    print(f"  {'oof/semilla':<13} {r.dif_oof:.3g}")
    print(f"  {'coeficientes':<13} {r.dif_coeficientes:.3g}")
    media = r.metricas[["roc_auc", "pr_auc", "boyce"]].mean()
    print(f"\nMedia: ROC-AUC {media['roc_auc']:.3f} · PR-AUC {media['pr_auc']:.3f} · Boyce {media['boyce']:.3f}")
    if r.identica:
        print("REPRODUCIDO: idéntico al paquete")
    elif r.ok(a.tol):
        print(f"REPRODUCIDO dentro de la tolerancia {a.tol:g}")
    else:
        print(f"NO REPRODUCIDO (tolerancia {a.tol:g})")
    return 0 if r.ok(a.tol) else 1


if __name__ == "__main__":
    sys.exit(main())
