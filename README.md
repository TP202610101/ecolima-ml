# ecolima-ml

Pipeline de Machine Learning para recomendar ubicaciones óptimas de puntos
de reciclaje en Lima Metropolitana.

Componente ML del proyecto EcoLima — Taller de Proyecto I (TP1), 2026-10.
Universidad Peruana de Ciencias Aplicadas (UPC).

**Autora:** Nikole García Chávez — Ciencias de la Computación  
**Co-autor:** Alexander Cantoral Sedamano — Ingeniería de Software  
**Asesora metodológica:** Esther Aliaga Cerna  
**Co-asesor:** Peter Jonathan Montalvo Garcia

---

## Descripción

EcoLima ML analiza datos geoespaciales (OpenStreetMap), demográficos (INEI),
socioeconómicos (APEIM) y operativos (puntos de reciclaje reales) para
identificar zonas óptimas donde instalar contenedores de reciclaje en los
50 distritos de Lima Metropolitana.

El modelo (LightGBM + SHAP) opera sobre una grilla espacial de celdas de
500m × 500m (~11,000 celdas sobre Lima). Cada celda recibe un score de
idoneidad (0–1) con explicaciones por variable vía SHAP values.

---

## Stack

| Componente | Tecnología |
|---|---|
| Algoritmo ML | LightGBM |
| Interpretabilidad | SHAP (TreeExplainer) |
| Pipeline | Python 3.11 |
| Datos geoespaciales | osmnx, geopandas, PostGIS |
| Experimentos | Google Colab / notebooks locales |
| Serialización | joblib |

---

## Estructura del repositorioecolima-ml/

```
├── data/
│   ├── raw/              # Datos originales (no versionados)
│   ├── processed/        # Features engineered (no versionados)
│   ├── simulated/        # Dataset EMI simulado para desarrollo
│   └── recycling_points/ # Puntos reales: Miraflores + Magdalena del Mar
├── src/
│   └── files/
│       ├── config.py
│       ├── data_generator.py
│       ├── evaluator.py
│       ├── explainer.py
│       ├── predictor.py
│       ├── preprocessing.py
│       ├── train_pipeline.py
│       └── trainer.py
├── notebooks/
│   ├── 01_eda_simulated_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_benchmarking.ipynb
│   └── 04_shap_analysis.ipynb
├── models/               # Modelos serializados (no versionados)
├── outputs/
│   ├── metrics/
│   ├── shap_plots/
│   └── recommendations/
└── docs/
├── decisions/
│   ├── ADR-001-lightgbm-over-xgboost.md
│   ├── ADR-002-clasificacion-binaria-vs-regresion.md
│   └── ADR-003-spatial-cv-strategy.md
├── challenges/
│   └── data-gaps.md
└── pipeline-stages.md

```


---

## Cómo reproducir

> Los datos crudos no están versionados. Ver `docs/challenges/data-gaps.md`
> para el estado actual de disponibilidad de datos.

```bashpip install -r requirements.txt
python src/files/train_pipeline.py

Para desarrollo con datos simulados:

```bashpython src/files/data_generator.py   # genera dataset EMI simulado
python src/files/train_pipeline.py

---

## Estado actual

Ver `CHANGELOG.md` para el historial detallado.

| Etapa | Estado |
|---|---|
| Generación de datos simulados | ✅ Completo |
| Preprocesamiento | ✅ Completo |
| Entrenamiento LightGBM | ✅ Completo |
| Evaluación de métricas | ✅ Completo |
| Interpretabilidad SHAP | ✅ Completo |
| Predicción sobre zonas candidatas | ✅ Completo |
| Datos reales (más allá de 2 distritos) | ⏳ En progreso |
| Spatial cross-validation | ⏳ Pendiente |
| Grilla 500m × 500m Lima completa | ⏳ Pendiente |

---

## Limitaciones conocidas

- Cobertura de entrenamiento actual: Miraflores y Magdalena del Mar (datos reales).
  Expansión progresiva en curso.
- Dataset EMI simulado usado para desarrollo inicial; debe reemplazarse con
  datos reales MINAM/municipales.
- Split espacial train/test aún no implementado por distrito (pendiente con
  datos de más distritos).

Ver `docs/challenges/data-gaps.md` para detalle completo.

---

## Referencias clave

- Valavi et al. (2019). blockCV: An r package for generating spatially or
  environmentally separated folds for k-fold cross-validation of species
  distribution models. *Methods in Ecology and Evolution*.
- Meyer & Pebesma (2021). Predicting into unknown space? Estimating the area
  of applicability of spatial prediction models. *Methods in Ecology and Evolution*.
- Koldasbayeva et al. (2024). Challenges in data-driven geospatial modeling
  for environmental research.


  el minimo de precision debe ser 80%

  