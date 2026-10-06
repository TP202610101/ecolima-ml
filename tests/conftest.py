# Fecha de creación: 2026-10-03
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecolima_ml.config import RUTA_PAQUETE  # noqa: E402
from ecolima_ml.paquete import sha256_archivo  # noqa: E402


@pytest.fixture(scope="session")
def ruta_paquete() -> Path:
    return RUTA_PAQUETE


@pytest.fixture
def copia_paquete(tmp_path) -> Path:
    """Copia del paquete en un directorio temporal, para corromperla sin tocar el original."""
    destino = tmp_path / "paquete"
    shutil.copytree(RUTA_PAQUETE, destino)
    return destino


def editar_csv(paq: Path, ruta: str, fn) -> None:
    """Modifica un CSV del paquete conservando los valores como texto (fn recibe y devuelve el DataFrame)."""
    df = pd.read_csv(paq / ruta, dtype=str, keep_default_na=False)
    df = fn(df)
    df.to_csv(paq / ruta, index=False, lineterminator="\n")


def resellar(paq: Path) -> None:
    """Actualiza sha256 y bytes del manifiesto para que solo falle la validación que se está probando."""
    man_path = paq / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for a in man["archivos"]:
        f = paq / a["ruta"]
        a["sha256"], a["bytes"] = sha256_archivo(f), f.stat().st_size
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
