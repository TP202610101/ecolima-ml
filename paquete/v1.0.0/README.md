<!-- Fecha de creación: 2026-10-03 -->
<!-- Fecha de modificación: 2026-10-03 -->

# EcoLima — paquete de resultados v1.0.0

Propensión de emplazamiento de puntos de reciclaje públicos, fijos y multimaterial y priorización multicriterio
(MCDA) de sitios candidatos. Alcance: los 43 distritos de la provincia de Lima; 40 de ellos tienen
sitios candidatos en el dataset (14 con puntos de reciclaje existentes).

## Qué es y qué no es

- **Propensión**: cuánto se parece un sitio a los lugares donde hoy existen puntos de reciclaje
  (aprendizaje presencia–fondo). **No es idoneidad, viabilidad ni probabilidad de éxito.**
- `apto_tamizaje`: el sitio pasa las reglas evaluables con datos secundarios. **Ningún sitio es `viable`**:
  los controles C02–C09 quedan `pendiente_campo` hasta una verificación municipal o de campo.
- La MCDA ordena oportunidades de tamizaje; el top-110 debe leerse como zonas prioritarias.

## Contenido

| carpeta | archivo | contenido |
|---|---|---|
| `datos/` | `dataset_v1.0.0.csv` | 2 242 sitios × 15 variables (congelado) |
| `datos/` | `fuentes.csv` | fuentes con URL, licencia, atribución, fecha de corte y sha256 |
| `datos/` | `distritos.geojson` | límites de los 43 distritos de la provincia de Lima (EPSG:4326, `MultiPolygon`) |
| `docs/` | `features.md` | definición de las 15 variables |
| `modelo/` | `logistica.joblib` | modelo final (scikit-learn 1.9.1) |
| `modelo/` | `logistica_parametros.json` | el mismo modelo en parámetros portables (sin pickle) |
| `modelo/` | `model_card.json` | métricas, comparación con LightGBM y límites |
| `modelo/` | `metricas_por_semilla.csv` | métricas fuera de fold por semilla (referencia de verificación) |
| `salidas/` | `sitios.csv`, `features.csv` | sitios y variables por sitio |
| `salidas/` | `tamizaje.csv` | reglas T01–T05 y estado por sitio |
| `salidas/` | `propension.csv` | propensión fuera de fold, propensión final, percentil y contribuciones por grupo de variables |
| `salidas/` | `propension_oof_por_semilla.csv` | propensión fuera de fold de cada semilla |
| `salidas/` | `mcda.csv`, `mcda_montecarlo.csv`, `criterios_mcda.csv`, `pesos_mcda.csv`, `concordancia_mcda.csv` | MCDA v1.0.2 (`mcda_montecarlo.csv`: una fila por sitio) |
| | `LICENCIA-DATOS.md` | licencia de los datos y avisos de cada fuente |
| | `manifest.json` | versiones, sha256 de cada archivo, librerías, semillas y métricas de referencia |

## Modelo de referencia: regresión logística

Validación cruzada agrupada por distrito (5 folds, 10 semillas), predicciones fuera de fold; media
[P2,5–P97,5] entre semillas. Prevalencia 0.0210.

| ROC-AUC | PR-AUC | Boyce | recall@10 % | recall@20 % |
|---|---|---|---|---|
| 0.749 [0.744–0.752] | 0.072 [0.064–0.076] | 0.757 [0.728–0.802] | 0.426 [0.426–0.426] | 0.515 [0.489–0.570] |

LightGBM (por defecto y ajustado con CV anidada) no la supera; se documenta como comparación en `model_card.json`.

Para mostrar la propensión, usar `percentil_oof`: el valor crudo no es una probabilidad (`class_weight='balanced'`).
Las contribuciones se publican por grupo de variables (clase, población, vías, POIs, construido, pendiente,
generación per cápita): los radios de una misma variable son colineales y sus coeficientes individuales tienen
signos opuestos que se compensan, así que no se interpretan por separado.

## MCDA v1.0.2

WLC sobre los sitios `apto_tamizaje` con cuatro criterios de beneficio normalizados min–max: demanda,
generación potencial, brecha de cobertura y densidad vial. Pesos iguales (principal), CRITIC y entropía
(contraste), y 5000 réplicas Monte Carlo de pesos (Dirichlet uniforme).

**La propensión no entra en el puntaje**; se entrega como contexto (`contexto_propension_oof` en
`criterios_mcda.csv`, percentil en `propension.csv`). Motivos: la propensión mide parecido con los lugares donde
ya hay puntos, lo que contradice el criterio de brecha de cobertura y no expresa conveniencia; con peso, el
ranking dependía del modelo que la genera y perdía estabilidad. Como escenario de sensibilidad, añadirla con
peso igual deja Spearman 0.847 y 31% del top-110 compartido.

Estabilidad: pesos iguales frente a CRITIC, Spearman 0.997 y 97% del
top-110 compartido; 5 sitios están en el top-110 en al menos 90 % de las réplicas Monte Carlo.

## Cómo verificar

1. Recalcular el sha256 de cada archivo y compararlo con `manifest.json`.
2. Reentrenar la logística con `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)`,
   grupos = `ubigeo`, semillas 0–9 y las versiones de `manifest.json`; las métricas por semilla deben
   coincidir con `modelo/metricas_por_semilla.csv`.

## Licencia

Los datos de este paquete forman una base derivada de OpenStreetMap y se ponen a disposición bajo la Open
Database License 1.0 (https://opendatacommons.org/licenses/odbl/1-0/). Contienen además información de
WorldPop y GHSL (CC BY 4.0), Copernicus DEM, MINAM-SIGERSOL e IGN (ODC-By) y de la Municipalidad
Distrital de San Isidro; los avisos completos de cada fuente están en `datos/fuentes.csv`.

El texto de los avisos por fuente y lo que cubre la licencia están en `LICENCIA-DATOS.md`.

## Atribución

Contiene datos de © OpenStreetMap contributors (ODbL 1.0), WorldPop (CC BY 4.0), GHSL del JRC (CC BY 4.0),
Copernicus DEM (licencia Copernicus), MINAM-SIGERSOL (ODC-By) y los límites distritales del IGN, de origen INEI
(ODC-By, https://opendatacommons.org/licenses/by/1-0/). Las estaciones ER01–ER10 provienen del Plan Anual de
Trabajo 2026 «Recicla San Isidro» de la Municipalidad Distrital de San Isidro (documento oficial sin licencia
explícita; se cita la fuente). Textos completos en `datos/fuentes.csv`.
