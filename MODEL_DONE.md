# ¿Qué significa "terminar el modelo"?

> Este archivo define exactamente qué hay que hacer, en qué orden,
> y cómo se ve el estado "terminado" para poder escribir el paper.
> Repo: `ecolima-ml`

---

## Estado actual

| Componente | Estado |
|---|---|
| Pipeline end-to-end | ✅ Funciona (validado en run_001 sintético y run_002 real) |
| Datos sintéticos | ✅ Genera y entrena — AUC 0.54 |
| Datos reales | ⚠ Integrados — `dataset_entrenamiento_v1.csv` (54 filas, 2 distritos) — AUC 0.50 |
| Hiperparámetros | ⚠ HPO corre (50 trials Optuna) — no mejora porque faltan datos |
| SHAP | ✅ `explainer.py` existe — no validado con datos reales |
| Métricas finales | ❌ Pendiente — AUC actual 0.50, necesita ≥ 0.70 |
| Modelo serializado final | ⚠ `lgbm_recycling.joblib` existe — entrenado con datos reales pero sin señal útil |

**El pipeline corre con datos reales pero el modelo no aprende todavía.**
Causa raíz: solo 4 positivos en train (distrito 1). Necesitas más datos balanceados.

---

## Qué hacer ahora — en orden

### 1. Armar el dataset real `(hoy)`

El cuello de botella es este paso. Sin datos reales, nada de lo que sigue tiene sentido.

- [ ] Tomar los puntos georreferenciados de Miraflores (31) y Magdalena del Mar (22) — total 53 puntos reales
- [ ] Definir las zonas candidatas negativas: celdas/zonas del mismo territorio sin punto de reciclaje (generar con `data_generator.py` modificado o manualmente con OSM)
- [ ] **Decidir el ratio positivos/negativos** — recomendado: 1:2 o 1:3 (53 positivos → ~100–160 negativos) para no repetir el desbalance del run_001
- [ ] Para cada zona (positiva y negativa) calcular las features:
  - Densidad poblacional (zona censal INEI)
  - NSE predominante (APEIM)
  - Distancia al nodo vial más cercano (OSM / PostGIS `ST_Distance`)
  - Densidad de POIs en radio 500m (OSM `ST_DWithin`)
  - Distancia al punto de reciclaje existente más cercano
  - Cobertura actual en radio 1km (¿cuántos puntos ya hay cerca?)
- [ ] Guardar como `data/real/miraflores_magdalena_v1.csv`
- [ ] Verificar que no haya leakage: las features de una zona positiva no deben incluir ese mismo punto como referencia

### 2. Entrenamiento con datos reales

- [x] Correr `train_pipeline.py` apuntando al CSV real — **corre sin errores**
- [x] HPO con Optuna funciona (cae a StratifiedKFold cuando hay 1 solo distrito en train)
- [ ] **Aumentar positivos en train a ≥ 15** (ahora hay 4 — insuficiente)
- [ ] **Balancear negativos por distrito** (ratio 1:2 ó 1:3 en cada distrito, no global)
- [ ] Re-optimizar hiperparámetros con Optuna una vez que haya datos suficientes
- [ ] Spatial CV: leave-one-district-out (ya implementado — necesita ≥ 2 distritos bien balanceados)
- [ ] Guardar como `run_002_real_miraflores_magdalena.md` en `outputs/metrics/`

### 3. Validar las métricas

El modelo está "terminado" cuando alcanza esto como mínimo:

| Métrica | Umbral mínimo aceptable | Umbral bueno para el paper |
|---|---|---|
| AUC-ROC | > 0.70 | > 0.75 |
| F1 (con umbral calibrado) | > 0.60 | > 0.65 |
| Precision | > 0.55 | > 0.65 |
| Recall | > 0.60 | > 0.70 |

Si el AUC queda entre 0.65–0.70, **igual es publicable** — documéntalo como limitación de dataset pequeño (53 puntos reales) y menciona que el pipeline está listo para escalar con más datos.

### 4. SHAP — interpretabilidad

- [ ] Correr `explainer.py` con el modelo entrenado en datos reales
- [ ] Generar SHAP summary plot (beeswarm) — captura para el paper
- [ ] Identificar las top-3 features más influyentes en las predicciones
- [ ] Escribir una oración interpretando el resultado: "La zona se recomienda principalmente porque X (+0.XX) y Y (+0.XX), penalizada por Z (-0.XX)"
- [ ] Guardar plots en `outputs/shap/`

### 5. Serializar el modelo final

- [ ] Guardar `lgbm_recycling_real_v1.joblib` en `models/`
- [ ] Actualizar `lgbm_recycling_metadata.json` con: fecha, datos usados, métricas finales, hiperparámetros

---

## Cómo se ve "terminado"

El modelo está terminado cuando puedes completar esta tabla sin dejar celdas vacías:

```
Dataset real: data/real/miraflores_magdalena_v1.csv
Filas totales: ___  (positivos: ___ / negativos: ___)
Split: leave-one-district-out (Miraflores / Magdalena)

AUC-ROC:   ___
F1:        ___
Precision: ___
Recall:    ___

Top feature 1: ___ (SHAP: +___)
Top feature 2: ___ (SHAP: +___)
Top feature 3: ___ (SHAP: -___)

Modelo serializado: models/lgbm_recycling_real_v1.joblib ✅
SHAP plot guardado: outputs/shap/summary_plot_v1.png    ✅
Run documentado:    outputs/metrics/run_002_real_*.md   ✅
```

Cuando esa tabla está llena → el modelo está terminado → puedes escribir
la sección de resultados del paper.

---

## Lo que NO necesitas para "terminar el modelo" hoy

No te traben estas cosas que pueden venir después:

- ❌ Integrar el modelo al API (eso es OE3, no OE2)
- ❌ Conectar con la base de datos PostGIS (idem)
- ❌ Validar con más de 2 distritos
- ❌ Spatial CV sofisticado — leave-one-district-out es suficiente para 2 distritos
- ❌ Resultados perfectos — AUC > 0.70 con 53 puntos reales ya es un resultado honesto y publicable

---

## Sobre los experimentos en el repo

Guarda **todo** en `outputs/metrics/` — sintéticos, intermedios y finales.
Para saber qué citar en el paper, agrega esta columna a la tabla del README:

| Run | Datos | AUC-ROC | F1 | Estado | Paper |
|---|---|---|---|---|---|
| 001 | Sintético (~400 filas) | 0.5426 | 0.75* | ⚠ sin señal real | ✅ (validación pipeline) |
| 002 | Real Mir+Mag (54 filas, 4 pos en train) | 0.5000 | 0.00 | 🔴 aleatorio — faltan datos | ❌ |
| 003 | Real Mir+Mag ampliado (pendiente) | — | — | pendiente | — |

La columna "Paper" marca si ese run aparece en el paper. El run_001 sí aparece
(como validación de que el pipeline funciona, no como resultado del modelo).
