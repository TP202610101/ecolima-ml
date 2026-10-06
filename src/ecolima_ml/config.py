# Fecha de creación: 2026-10-03
"""Versiones soportadas y rutas por defecto."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RUTA_PAQUETE = ROOT_DIR / "paquete" / "v1.0.0"

# Versiones que este código sabe leer (contrato del paquete v1, §5). Un paquete con otra
# versión mayor se rechaza; uno con otra versión de modelo o MCDA también, porque cambian
# features, fórmula o criterios.
VERSIONES = {
    "paquete_mayor": 1,
    "dataset": "1.0.0",
    "modelo": "propension-1.0.0",
    "mcda": "1.0.2",
}

AVISO_PROPENSION = (
    "Propensión de emplazamiento: parecido del sitio con los lugares donde ya existen puntos de reciclaje. "
    "No es idoneidad, viabilidad ni probabilidad calibrada; para mostrarla se usa el percentil."
)
