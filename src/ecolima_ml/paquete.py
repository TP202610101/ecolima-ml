# Fecha de creación: 2026-10-03
"""Carga del paquete de resultados: manifiesto, sha256 y compatibilidad de versiones."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ecolima_ml.config import RUTA_PAQUETE, VERSIONES


class ErrorPaquete(Exception):
    """El paquete no existe, no es compatible o no coincide con su manifiesto."""


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


class Paquete:
    """Paquete de resultados versionado (contrato del paquete v1)."""

    def __init__(self, ruta: Path | str = RUTA_PAQUETE):
        self.ruta = Path(ruta)
        archivo = self.ruta / "manifest.json"
        if not archivo.is_file():
            raise ErrorPaquete(f"no existe manifest.json en {self.ruta}")
        self.manifest = json.loads(archivo.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ versiones
    @property
    def versiones(self) -> dict[str, str]:
        m = self.manifest
        return {"paquete": m["paquete"], "dataset": m["dataset"]["version"], "modelo": m["modelo"]["version"],
                "mcda": m["mcda"]["version"]}

    def errores_version(self) -> list[str]:
        v = self.versiones
        e = []
        if int(v["paquete"].split(".")[0]) != VERSIONES["paquete_mayor"]:
            e.append(f"paquete {v['paquete']}: se espera la versión mayor {VERSIONES['paquete_mayor']}")
        for k in ("dataset", "modelo", "mcda"):
            if v[k] != VERSIONES[k]:
                e.append(f"{k} {v[k]}: este código soporta {VERSIONES[k]}")
        return e

    def comprobar_version(self) -> None:
        e = self.errores_version()
        if e:
            raise ErrorPaquete("versión no soportada: " + "; ".join(e))

    # ------------------------------------------------------------------ integridad
    def errores_hash(self) -> list[str]:
        """sha256 y bytes de cada archivo frente al manifiesto; ningún archivo sin listar (validación 1)."""
        listados = {a["ruta"]: a for a in self.manifest["archivos"]}
        reales = {p.relative_to(self.ruta).as_posix() for p in self.ruta.rglob("*") if p.is_file()} - {"manifest.json"}
        e = [f"sin listar en el manifiesto: {r}" for r in sorted(reales - set(listados))]
        e += [f"listado pero ausente: {r}" for r in sorted(set(listados) - reales)]
        for r in sorted(reales & set(listados)):
            if (self.ruta / r).stat().st_size != listados[r]["bytes"]:
                e.append(f"bytes distinto: {r}")
            elif sha256_archivo(self.ruta / r) != listados[r]["sha256"]:
                e.append(f"sha256 distinto: {r}")
        return e

    def comprobar_hashes(self) -> None:
        e = self.errores_hash()
        if e:
            raise ErrorPaquete("integridad: " + "; ".join(e[:5]))

    # ------------------------------------------------------------------ lectura
    def archivo(self, ruta: str) -> Path:
        return self.ruta / ruta

    def json(self, ruta: str) -> dict:
        return json.loads(self.archivo(ruta).read_text(encoding="utf-8"))

    def csv(self, ruta: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.archivo(ruta), **kwargs)

    def dataset(self) -> pd.DataFrame:
        """Dataset congelado; ubigeo como texto, igual que en el entrenamiento validado."""
        return self.csv(self.manifest["dataset"]["archivo"], dtype={"ubigeo": str})


def cargar(ruta: Path | str = RUTA_PAQUETE, verificar: bool = True) -> Paquete:
    """Abre un paquete y, por defecto, exige versión soportada y hashes correctos."""
    p = Paquete(ruta)
    if verificar:
        p.comprobar_version()
        p.comprobar_hashes()
    return p
