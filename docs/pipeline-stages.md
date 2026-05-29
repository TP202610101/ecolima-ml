# Etapas del Pipeline ML

Descripción narrativa de cada etapa: qué hace, qué decisiones involucra,
qué produce, y qué problemas se encontraron.

---

## Etapa 1: Ingesta de datos

**Módulo:** `data_generator.py`  
**Entrada:** Fuentes externas (INEI, APEIM, OSM, puntos reales)  
**Salida:** CSVs crudos en `data/raw/` y dataset simulado en `data/simulated/`

**Descripción:**
Consolida datos de múltiples fuentes heterogéneas en un formato unificado.
Durante el desarrollo inicial se usa el dataset EMI simulado (200–300 filas
con valores representativos de Lima) mientras se completa la recolección de
datos reales.

Para OSM se usa `osmnx`:
```python
import osmnx as ox
G = ox.graph_from_place("Miraflores, Lima, Peru", network_type="walk")
```

**Decisiones tomadas:**
- Proyección base: EPSG:4326 para almacenamiento; conversión a EPSG:32718
  (UTM 18S) para cálculos de distancia en metros
- Nombres de distritos estandarizados según UBIGEO INEI

**Problemas encontrados:**
- Ver `docs/challenges/data-gaps.md`

---

## Etapa 2: Preprocesamiento

**Módulo:** `preprocessing.py`  
**Entrada:** `data/raw/`  
**Salida:** `data/processed/` — features limpias y normalizadas

**Descripción:**
- Imputación de nulos: mediana para variables numéricas
- Deduplicación de puntos de reciclaje por coordenada
- Estandarización de nombres de distritos
- Normalización de coordenadas al sistema de referencia del proyecto

**Decisiones tomadas:**
- Mediana sobre media para imputación: más robusta ante outliers en
  variables de densidad urbana (distribución sesgada en Lima)
- No se aplica escalado de features: LightGBM es invariante a escala

---

## Etapa 3: Feature Engineering

**Módulo:** `preprocessing.py` (sección feature engineering)  
**Entrada:** `data/processed/`  
**Salida:** Feature matrix lista para entrenamiento

**Features calculadas:**

| Feature | Cálculo | Fuente |
|---|---|---|
| `density_recycling_1km` | Conteo de puntos en buffer 1km | PostGIS ST_DWithin |
| `dist_nearest_point` | Distancia al punto más cercano | PostGIS ST_Distance |
| `dist_nearest_road` | Distancia al nodo vial principal | osmnx + ST_Distance |
| `poi_density_500m` | Conteo de POIs en radio 500m | OSM Overpass |
| `walkability_score` | Isócrona 5 min a pie | osmnx network analysis |
| `coverage_gap_index` | Población sin punto en 500m / población total | INEI + PostGIS |
| `nse_ab_pct` | % NSE A+B en zona | APEIM |
| `population_density` | Hab/km² | INEI |

**Decisiones tomadas:**
- Radio de 500m para densidad de POIs: basado en literatura de
  accesibilidad peatonal urbana (5–7 min caminando)
- `coverage_gap_index` como feature derivada de alta relevancia:
  captura zonas pobladas sin cobertura actual

---

## Etapa 4: Entrenamiento

**Módulo:** `trainer.py`, `train_pipeline.py`  
**Entrada:** Feature matrix procesada  
**Salida:** Modelo serializado en `models/lgbm_model.joblib`

**Descripción:**
Entrenamiento de LightGBM con búsqueda de hiperparámetros via Optuna
o GridSearch. Validación cruzada k=5 (o spatial CV cuando haya datos
suficientes).

**Hiperparámetros clave monitoreados:**
- `num_leaves`: controla complejidad del árbol
- `min_data_in_leaf`: previene overfitting en dataset pequeño
- `lambda_l1`, `lambda_l2`: regularización
- `learning_rate` + `n_estimators`: con early stopping

**Decisiones tomadas:**
- Split por distrito (no aleatorio) para evitar data leakage espacial
- `class_weight='balanced'` o `scale_pos_weight` por desbalance de clases
  (pocas celdas positivas vs miles de negativas)

**Ver:** ADR-001, ADR-002, ADR-003

---

## Etapa 5: Evaluación

**Módulo:** `evaluator.py`  
**Entrada:** Modelo entrenado + test set  
**Salida:** Métricas en `outputs/metrics/`

**Métricas reportadas:**
- F1-score (macro y weighted)
- Precision, Recall por clase
- AUC-ROC
- Matriz de confusión

**Criterio de éxito mínimo (propuesta):**
- F1-score ≥ 0.70 en test set
- AUC-ROC ≥ 0.75

---

## Etapa 6: Interpretabilidad SHAP

**Módulo:** `explainer.py`  
**Entrada:** Modelo entrenado + instancias a explicar  
**Salida:** SHAP values en `outputs/shap_plots/`

**Descripción:**
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

Produce para cada zona candidata: top-5 features con mayor contribución
positiva y negativa a la predicción. Este output es el que consume el
endpoint `POST /predict` del backend FastAPI.

---

## Etapa 7: Predicción sobre zonas candidatas

**Módulo:** `predictor.py`  
**Entrada:** Grilla de ~11,000 celdas de 500m × 500m sobre Lima  
**Salida:** Ranking de zonas por score + SHAP top features en `outputs/recommendations/`

**Estado actual:** Pendiente — requiere grilla completa de Lima y datos
de más distritos para ser representativo.