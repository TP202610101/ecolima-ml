# Fecha de creación: 2026-10-03
"""Cada validación del contrato debe aprobar el paquete oficial y detectar su propio caso de corrupción."""

import json

import pytest

from conftest import editar_csv, resellar
from ecolima_ml.validacion import validar


def fallan(paq) -> set[int]:
    return {r.n for r in validar(paq) if not r.ok}


def test_paquete_oficial_pasa_las_15(ruta_paquete):
    resultados = validar(ruta_paquete)
    assert [r.n for r in resultados] == list(range(1, 16))
    assert all(r.ok for r in resultados), [(r.n, r.errores) for r in resultados if not r.ok]


def _fila(df, col, valor, i=0):
    df.loc[i, col] = valor
    return df


def _fila_tope(df):
    i = df.index[(df["esquema"] == "iguales") & (df["rango"] == "1")][0]
    df.loc[i, "en_top_k"] = "False"
    return df


def _sitio_apto(paq):
    import pandas as pd
    t = pd.read_csv(paq / "salidas/tamizaje.csv", dtype=str, keep_default_na=False)
    return t.loc[t["estado_tamizaje"] == "apto_tamizaje"].index[0]


CASOS = {
    2: ("salidas/mcda_montecarlo.csv", lambda df: _fila(df, "mcda_version", "1.0.1")),
    3: ("datos/fuentes.csv", lambda df: _fila(df, "licencia_estado", "no_verificada")),
    4: ("salidas/tamizaje.csv", lambda df: df[[*df.columns[:5], "T05_sin_punto_existente_30m",
                                               *[c for c in df.columns[5:] if c != "T05_sin_punto_existente_30m"]]]),
    6: ("salidas/features.csv", lambda df: df.drop(index=0)),
    7: ("salidas/propension.csv", lambda df: _fila(df, "rol", "positivo" if df.loc[0, "rol"] == "fondo" else "fondo")),
    8: ("salidas/mcda_montecarlo.csv", lambda df: df.drop(index=0)),
    9: ("salidas/features.csv", lambda df: _fila(df, "pob_wp_r500", repr(float(df.loc[0, "pob_wp_r500"]) * (1 + 1e-6)))),
    10: ("salidas/propension_oof_por_semilla.csv", lambda df: _fila(df, "semilla_0", "0.5")),
    11: ("salidas/propension.csv", lambda df: _fila(df, "percentil_oof", "0.123")),
    12: ("salidas/propension.csv", lambda df: _fila(df, "contrib_vias", repr(float(df.loc[0, "contrib_vias"]) + 0.01))),
    13: ("salidas/mcda.csv", _fila_tope),
    14: ("salidas/criterios_mcda.csv", lambda df: _fila(df, "K1_demanda_norm", "1.5")),
    15: ("salidas/sitios.csv", lambda df: _fila(df, "lon", "-80.0")),
}


def test_control_reescribir_sin_cambios_no_falla(copia_paquete):
    """El procedimiento de corrupción (reescribir y resellar) no provoca fallos por sí mismo."""
    for ruta, _ in CASOS.values():
        editar_csv(copia_paquete, ruta, lambda df: df)
    resellar(copia_paquete)
    assert fallan(copia_paquete) == set()


@pytest.mark.parametrize("n", sorted(CASOS))
def test_cada_validacion_detecta_su_caso(copia_paquete, n):
    ruta, fn = CASOS[n]
    editar_csv(copia_paquete, ruta, fn)
    resellar(copia_paquete)
    assert n in fallan(copia_paquete)


def test_1_detecta_hash_sin_resellar(copia_paquete):
    editar_csv(copia_paquete, "salidas/sitios.csv", lambda df: _fila(df, "distrito", "OTRO"))
    assert 1 in fallan(copia_paquete)


def test_5_detecta_viable_y_valores_fuera_de_enumeracion(copia_paquete):
    i = _sitio_apto(copia_paquete)
    editar_csv(copia_paquete, "salidas/tamizaje.csv", lambda df: _fila(df, "estado_tamizaje", "viable", i))
    resellar(copia_paquete)
    r = {x.n: x for x in validar(copia_paquete)}
    assert not r[5].ok and any("viable" in e for e in r[5].errores)


def test_5_detecta_columna_viable(copia_paquete):
    editar_csv(copia_paquete, "salidas/concordancia_mcda.csv", lambda df: _fila(df, "comparacion", "es viable"))
    resellar(copia_paquete)
    assert 5 in fallan(copia_paquete)


def test_4_detecta_geojson_sin_multipolygon(copia_paquete):
    f = copia_paquete / "datos/distritos.geojson"
    g = json.loads(f.read_text(encoding="utf-8"))
    g["features"][0]["geometry"]["type"] = "Polygon"
    f.write_text(json.dumps(g), encoding="utf-8")
    resellar(copia_paquete)
    assert 4 in fallan(copia_paquete)


def test_archivo_ausente_cuenta_como_fallo(copia_paquete):
    (copia_paquete / "salidas/mcda.csv").unlink()
    f = fallan(copia_paquete)
    assert {1, 2, 4, 8, 13} <= f
