<!-- Fecha de modificación: 2026-10-06 -->

# ecolima-ml

Componente de Machine Learning de EcoLima: **propensión de emplazamiento** de puntos de reciclaje públicos en los
43 distritos de la provincia de Lima, y entrega versionada de los resultados (tamizaje, propensión y priorización
MCDA) al backend.

Proyecto EcoLima — Universidad Peruana de Ciencias Aplicadas (UPC).
**Autora:** Nikole García Chávez · **Coautor:** Alexander Cantoral Sedamano

## Qué contiene

```
ecolima-ml/
├── paquete/v1.0.0/     paquete de resultados que consume el backend (cada archivo con su sha256 en manifest.json)
│   ├── datos/          dataset congelado, fuentes y distritos
│   ├── modelo/         parámetros de la regresión logística, model card y métricas por semilla
│   └── salidas/        sitios, tamizaje, propensión y MCDA
├── src/ecolima_ml/     carga del paquete, modelo, validaciones y reproducción
├── scripts/            validar_paquete.py, reproducir_metricas.py
└── tests/
```

El backend usa solo `paquete/v1.0.0/`: lo importa, verifica los hashes y sirve los resultados sin recalcular. El
resto del repositorio permite comprobar que el paquete es correcto y reproducible.

## Versión actual

| Componente | Versión |
|---|---|
| Paquete | `1.0.0` |
| Dataset | `1.0.0` (2 242 sitios: 47 puntos existentes y 2 195 de fondo) |
| Modelo | `propension-1.0.0` (regresión logística, 15 variables) |
| MCDA | `1.0.2` (2 192 sitios `apto_tamizaje`) |

El contrato del paquete (archivos, columnas y validaciones) está en `contrato-paquete-v1.md`, repositorio `tp2-docs`.

## Cómo leer los resultados

- **Propensión:** cuánto se parece un sitio a los lugares donde hoy hay puntos de reciclaje. No es idoneidad,
  viabilidad ni probabilidad; se muestra como **percentil**.
- **`apto_tamizaje`:** el sitio pasa las reglas evaluables con datos secundarios. **Ningún sitio es «viable»**:
  no hay verificación de campo.
- **MCDA:** ordena los sitios aptos con cuatro criterios (demanda, generación, brecha y vías). El top-110 es
  «prioridad MCDA», no una recomendación final. La propensión no entra en el puntaje.

## Uso

```powershell
python -m pip install -r requirements.txt        # requirements-dev.txt añade pytest
python scripts/validar_paquete.py                # validaciones 1–15 del contrato (código 1 si falla)
python scripts/reproducir_metricas.py            # reentrena y compara con modelo/metricas_por_semilla.csv
python -m pytest
```

Propensión de una observación con las 15 variables:

```python
from ecolima_ml import ModeloPropension
from ecolima_ml.paquete import cargar

paquete = cargar()                                  # verifica versión y hashes
modelo = ModeloPropension.desde_paquete(paquete)
r = modelo.calcular(paquete.dataset().iloc[0].to_dict())
r.propension, r.contribuciones, r.aviso
```

## Limitaciones

- 47 positivos concentrados en 14 de 40 distritos; OSM mapea de forma desigual, así que las métricas son cotas
  inferiores.
- La propensión refleja dónde se instaló en el pasado, no dónde conviene instalar.
- Las variables llegan congeladas y no se recalculan desde este repositorio.

## Licencias

Los datos (`paquete/`) son una base derivada de OpenStreetMap bajo la **Open Database License 1.0**
(https://opendatacommons.org/licenses/odbl/1-0/), con datos de WorldPop, GHSL, Copernicus DEM, MINAM-SIGERSOL, IGN
y la Municipalidad Distrital de San Isidro. Avisos completos en `paquete/v1.0.0/LICENCIA-DATOS.md` y
`paquete/v1.0.0/datos/fuentes.csv`. Licencia del código: por decidir.

La implementación anterior (v0.3, datos simulados y LightGBM) quedó retirada; está en el historial de Git
(commit `9b98581`).
