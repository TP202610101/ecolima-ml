# Dataset real — estado actual y qué hacer hoy

> Última revisión: 2026-06-08
> Estado: `dataset_entrenamiento_v1.csv` existe y el pipeline corre end-to-end. AUC = 0.50 — el modelo no aprende todavía. Causa principal: solo 4 positivos en train.

---

## Qué CSV usa el modelo — estado actual

| Archivo | Filas | Positivos | Negativos | ¿Puede entrenar? |
|---------|-------|-----------|-----------|-----------------|
| `data/synthetic/synthetic_dataset.csv` | ~400 | ~200 | ~200 | ✅ Sí |
| `data/real/puntos_positivos_v1.csv` | 14 | 14 | 0 | ❌ No — sin negativos |
| `data/real/dataset_entrenamiento_v1.csv` | **54** | **14** | **40** | ✅ Carga y corre — pero datos insuficientes |

El pipeline ya usa el dataset real por defecto (`train_pipeline.py` sin argumentos).

---

## Resultado del primer run con datos reales (2026-06-08)

```
Train: 37 muestras (distrito 1)  — 4 positivos (10.8%), 33 negativos (89.2%)
Test:  17 muestras (distrito 2)  — 10 positivos (58.8%), 7 negativos (41.2%)

AUC-ROC : 0.5000  ← aleatorio
F1      : 0.0000
Precision: 0.0000
Recall  : 0.0000
Umbral  : 0.25

El modelo predice todo como clase 0.
```

---

## Por qué AUC = 0.5 — diagnóstico

| Causa | Severidad | Efecto |
|-------|-----------|--------|
| Solo **4 positivos en train** | 🔴 Crítico | LightGBM no puede aprender patrones de clase 1 |
| **Distribución muy distinta entre distritos**: train 10.8% pos vs test 58.8% pos | 🔴 Crítico | El test no representa el mismo problema que el train |
| Solo **1 distrito en train** → CV cae a StratifiedKFold(2) | 🟡 Moderado | No hay spatial CV real, métricas internas no son confiables |
| 54 filas totales (37 train) | 🟡 Moderado | LightGBM necesita al menos 100–200 muestras de train para regularizar |
| `poi_parks_500m` mapeada desde `has_park_300m` (proxy) | 🟢 Menor | Feature presente pero aproximada |

---

## Qué está presente en el dataset (columnas)

Columnas confirmadas en `dataset_entrenamiento_v1.csv` (35 columnas, 54 filas):

**Demográficas (OK):** `population_density`, `nse_ab_pct`, `nse_de_pct`, `nse_c_pct`, `num_households`, `urbanization_rate`, `household_size_avg`

**Geoespaciales (OK):** `dist_nearest_road_m`, `walkability_score`, `poi_commercial_500m`, `poi_educational_500m`, `land_use_encoded`, `area_m2`, `latitude`, `longitude`, `centroid_lat`, `centroid_lon`

**Operativas (OK):** `dist_nearest_recycling_m`, `recycling_density_1km`, `waste_per_capita_kg`

**Adicionales del notebook (descartadas por el modelo):** `income_stratum`, `fuel_expenditure_sol`, `edu_level_head`, `pct_recyclable`, `pct_plastic`, `informal_recyclers_count`, `slope_pct`, `road_density`, `dist_to_market_m`, `recycling_potential_index`

**Columnas que faltan de `config.py`:**
| Feature | Estado | Solución aplicada |
|---------|--------|-------------------|
| `poi_parks_500m` | ❌ No en CSV | ✅ `FeatureEngineer` la deriva de `has_park_300m` |
| `coverage_gap_index` | ❌ No en CSV | ✅ `FeatureEngineer` la calcula desde `recycling_deficit` |
| `accessibility_composite` | Engineered | ✅ Calculada automáticamente |
| `nse_high_ratio` | Engineered | ✅ Calculada automáticamente |
| `recycling_deficit` | Engineered | ✅ Calculada automáticamente |

---

## Qué falta y qué mejorar — prioridades

### Prioridad 1 (bloquea aprendizaje): más positivos en train

- El train tiene **4 positivos**. Necesitas mínimo **15–20** para que LightGBM generalice.
- Opciones:
  - **A) Agregar más puntos reales de Miraflores** (distrito 1 = train). Si tienes 31 puntos de Miraflores, todos deberían estar en el CSV — revisar por qué solo hay 4 positivos ahí.
  - **B) Agregar más negativos proporcionales** — si hay 4 pos, necesitas al menos 8–12 neg bien seleccionados (ratio 1:2 ó 1:3), no 33.

### Prioridad 2 (bloquea validación): distribución de clases por distrito inconsistente

- Distrito 1 (Miraflores): 10.8% positivos
- Distrito 2 (Magdalena): 58.8% positivos
- Esto indica que la generación de negativos no fue proporcional entre distritos.
- **Acción**: generar negativos de forma balanceada en ambos distritos (ej. 3× el número de positivos de cada distrito).

### Prioridad 3 (mejora modelo): agregar más distritos o más datos

- Con solo 2 distritos, el spatial CV no funciona correctamente.
- Si puedes agregar aunque sea **San Isidro o Barranco** (aunque sea con features estimadas), la CV mejora sustancialmente.

### Prioridad 4 (ajuste rápido): `class_weight='balanced'` en LightGBM

- Mientras no haya más datos, agregar `class_weight='balanced'` en `LGBM_BASE_PARAMS` en `config.py` puede ayudar con el desbalance de clases.
- Cambia: `"class_weight": "balanced"` en el dict `LGBM_BASE_PARAMS`.

---

## Estado del checklist

- [x] `puntos_positivos_v1.csv` generado (14 positivos con features)
- [x] Grid de negativos computado en notebook
- [x] `dataset_entrenamiento_v1.csv` guardado con columnas renombradas (54 filas)
- [x] `train_pipeline.py` corre sin error con datos reales
- [x] Features engineered resueltas (`poi_parks_500m`, `coverage_gap_index`, etc.)
- [ ] Balance corregido: train necesita ≥ 15 positivos y distribución similar al test
- [ ] AUC > 0.70 con datos reales
- [ ] Métricas finales registradas en `outputs/metrics/run_002_real_*.md`
