# Fecha de creación: 2026-10-03
import numpy as np
import pytest

from ecolima_ml.modelo import ErrorFeatures, ModeloPropension
from ecolima_ml.paquete import cargar


@pytest.fixture(scope="module")
def paquete(ruta_paquete):
    return cargar(ruta_paquete)


@pytest.fixture(scope="module")
def modelo(paquete):
    return ModeloPropension.desde_paquete(paquete)


@pytest.fixture(scope="module")
def dataset(paquete):
    return paquete.dataset()


@pytest.fixture
def observacion(dataset):
    return dataset.iloc[0].to_dict()


def test_features_oficiales(modelo):
    assert len(modelo.features) == 15
    assert modelo.categorica == "clase_oportunidad"
    # Ninguna feature se deriva del inventario de puntos existentes.
    assert not [f for f in modelo.features if "recycling" in f or "punto" in f or "dist_" in f]


def test_reproduce_propension_final_y_contribuciones(modelo, paquete, dataset):
    pub = paquete.csv("salidas/propension.csv").set_index("sitio_id").loc[dataset["sitio_id"]]
    assert np.abs(modelo.propension(dataset) - pub["propension_final"].to_numpy()).max() < 1e-12
    c = modelo.contribuciones(dataset)
    assert np.abs(c.to_numpy() - pub[[f"contrib_{g}" for g in c.columns]].to_numpy()).max() < 1e-12
    assert np.abs(modelo.logit(dataset) - modelo.logit_base - c.sum(axis=1).to_numpy()).max() < 1e-9


def test_resultado_individual_e_interpretacion(modelo, observacion):
    r = modelo.calcular(observacion)
    assert 0 < r.propension < 1
    assert r.modelo == "propension-1.0.0"
    assert set(r.contribuciones) == {"clase", "poblacion", "vias", "pois", "construido", "pendiente", "sigersol"}
    assert "No es idoneidad, viabilidad ni probabilidad" in r.aviso
    assert not hasattr(r, "is_optimal") and not hasattr(r, "is_recommended")


def test_feature_faltante(modelo, observacion):
    del observacion["pendiente_grados"]
    with pytest.raises(ErrorFeatures, match="pendiente_grados"):
        modelo.calcular(observacion)


@pytest.mark.parametrize("valor", ["alto", None, float("nan"), float("inf")])
def test_feature_numerica_invalida(modelo, observacion, valor):
    observacion["pob_wp_r500"] = valor
    with pytest.raises(ErrorFeatures):
        modelo.calcular(observacion)


def test_categoria_desconocida(modelo, observacion):
    observacion["clase_oportunidad"] = "estadio"
    with pytest.raises(ErrorFeatures, match="estadio"):
        modelo.calcular(observacion)
    # Regla explícita de los parámetros: todas las indicadoras en 0.
    r = modelo.calcular(observacion, permitir_desconocida=True)
    z = modelo.preparar(observacion, permitir_desconocida=True)
    assert (z[0, -len(modelo.categorias):] == 0).all()
    assert r.logit == pytest.approx(modelo.intercepto + z[0] @ modelo.coef, abs=1e-12)


def test_valores_negativos_se_recortan_antes_de_log1p(modelo, observacion):
    observacion["poi_n_km2_r250"] = -5.0
    a = modelo.calcular(observacion).propension
    observacion["poi_n_km2_r250"] = 0.0
    assert modelo.calcular(observacion).propension == a


def test_parametros_inconsistentes(paquete):
    params = paquete.json("modelo/logistica_parametros.json")
    params["coeficientes"] = params["coeficientes"][:-1]
    with pytest.raises(ValueError, match="dimensiones"):
        ModeloPropension(params)
