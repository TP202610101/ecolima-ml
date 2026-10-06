# Fecha de creación: 2026-10-03
import json

import pytest

from ecolima_ml.config import VERSIONES
from ecolima_ml.paquete import ErrorPaquete, cargar


def test_paquete_oficial_carga_con_hashes_y_versiones(ruta_paquete):
    p = cargar(ruta_paquete)
    assert p.versiones == {"paquete": "1.0.0", "dataset": VERSIONES["dataset"], "modelo": VERSIONES["modelo"],
                           "mcda": VERSIONES["mcda"]}
    ds = p.dataset()
    assert len(ds) == 2242
    assert (ds["rol"] == "positivo").sum() == 47
    assert ds["ubigeo"].nunique() == 40


def test_un_byte_cambiado_rompe_el_hash(copia_paquete):
    f = copia_paquete / "salidas/sitios.csv"
    datos = f.read_bytes()
    f.write_bytes(datos[:-2] + (b"X" if datos[-2:-1] != b"X" else b"Y") + datos[-1:])
    with pytest.raises(ErrorPaquete, match="sha256 distinto: salidas/sitios.csv"):
        cargar(copia_paquete)


def test_crlf_rompe_el_hash(copia_paquete):
    """Lo que haría Git con autocrlf si no existiera .gitattributes."""
    f = copia_paquete / "salidas/pesos_mcda.csv"
    f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ErrorPaquete, match="pesos_mcda.csv"):
        cargar(copia_paquete)


def test_archivo_sin_listar_y_archivo_ausente(copia_paquete):
    (copia_paquete / "salidas/extra.csv").write_text("a\n1\n", encoding="utf-8")
    (copia_paquete / "modelo/logistica.joblib").unlink()
    with pytest.raises(ErrorPaquete) as ex:
        cargar(copia_paquete)
    assert "sin listar en el manifiesto: salidas/extra.csv" in str(ex.value)
    assert "listado pero ausente: modelo/logistica.joblib" in str(ex.value)


@pytest.mark.parametrize("ruta,valor,mensaje", [
    (("mcda", "version"), "1.0.1", "mcda 1.0.1"),
    (("modelo", "version"), "propension-2.0.0", "modelo propension-2.0.0"),
    (("paquete",), "2.0.0", "paquete 2.0.0"),
])
def test_version_no_soportada(copia_paquete, ruta, valor, mensaje):
    man_path = copia_paquete / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    nodo = man
    for k in ruta[:-1]:
        nodo = nodo[k]
    nodo[ruta[-1]] = valor
    man_path.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ErrorPaquete, match=mensaje):
        cargar(copia_paquete)


def test_sin_manifiesto(tmp_path):
    with pytest.raises(ErrorPaquete, match="manifest.json"):
        cargar(tmp_path)
