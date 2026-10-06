#!/usr/bin/env python3
# Fecha de creación: 2026-10-03
"""Valida un paquete de resultados EcoLima contra el contrato del paquete v1 (§6, validaciones 1–15).

    python scripts/validar_paquete.py [--paquete paquete/v1.0.0] [--json]

Sale con código 0 si el paquete es compatible y pasa las 15 validaciones; 1 en otro caso.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecolima_ml.config import RUTA_PAQUETE  # noqa: E402
from ecolima_ml.paquete import ErrorPaquete, Paquete  # noqa: E402
from ecolima_ml.validacion import validar  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paquete", type=Path, default=RUTA_PAQUETE, help="carpeta del paquete (contiene manifest.json)")
    ap.add_argument("--json", action="store_true", help="salida en JSON")
    a = ap.parse_args()

    try:
        paq = Paquete(a.paquete)
    except ErrorPaquete as ex:
        print(f"RECHAZADO: {ex}", file=sys.stderr)
        return 1
    errores_version = paq.errores_version()
    resultados = validar(a.paquete)
    aprobado = not errores_version and all(r.ok for r in resultados)

    if a.json:
        print(json.dumps({"versiones": paq.versiones, "errores_version": errores_version, "aprobado": aprobado,
                          "validaciones": [{"n": r.n, "regla": r.regla, "ok": r.ok, "errores": r.errores[:5]}
                                           for r in resultados]}, ensure_ascii=False, indent=2))
    else:
        v = paq.versiones
        print(f"Paquete {v['paquete']} · dataset {v['dataset']} · modelo {v['modelo']} · MCDA {v['mcda']}")
        for e in errores_version:
            print(f"  VERSIÓN  {e}")
        for r in resultados:
            print(f"  {'OK   ' if r.ok else 'FALLA'} {r.n:>2}. {r.regla}")
            for e in r.errores[:5]:
                print(f"           - {e}")
        print("APROBADO" if aprobado else "RECHAZADO")
    return 0 if aprobado else 1


if __name__ == "__main__":
    sys.exit(main())
