# Run 001 — Baseline con datos sintéticos

**Fecha:** 2026-05-29
**Datos:** `data/synthetic/synthetic_dataset.csv` (dataset EMI simulado)
**Objetivo:** Validar que el pipeline end-to-end funciona. No es una
evaluación real del modelo.

---

## Parámetros (Optuna, 50 trials)

| Parámetro | Valor |
|---|---|
| num_leaves | 28 |
| learning_rate | 0.038 |
| n_estimators | 227 |
| min_child_samples | 56 |
| subsample | 0.703 |
| colsample_bytree | 0.865 |
| reg_alpha | 0.009 |
| reg_lambda | 0.036 |

---

## Métricas

| Métrica | Valor |
|---|---|
| CV AUC (train) | 0.5736 ± 0.0277 |
| Test AUC-ROC | 0.5426 |
| Test F1* | 0.7520 |
| Precision | 0.6025 |
| Recall | 1.0000 |
| Umbral óptimo | 0.10 |

*F1 engañoso — ver interpretación.

---

## Matriz de confusión

|  | Pred 0 | Pred 1 |
|---|---|---|
| Real 0 | 0 (TN) | 159 (FP) |
| Real 1 | 0 (FN) | 241 (TP) |

---

## Interpretación

**El modelo predice clase 1 para el 100% de las instancias.**

- AUC ≈ 0.54 indica capacidad discriminativa casi nula (aleatorio = 0.50)
- El umbral de 0.10 en vez de 0.50 confirma que las probabilidades están
  comprimidas — el modelo no separa bien las clases
- F1 = 0.75 es **artificialmente alto** por el desbalance: con 60% de
  positivos en test, predecir siempre 1 da F1 alto pero el modelo no
  aprende nada útil

---

## Causa identificada

El dataset sintético no genera suficiente señal discriminativa entre
zonas óptimas (is_optimal=1) y no óptimas (is_optimal=0). Las features
simuladas no reproducen la separabilidad espacial real de Lima.

Esto es un resultado esperado en esta etapa — el pipeline funciona
correctamente. El problema es la fuente de datos, no el modelo.

---

## Acciones siguientes

- [ ] Revisar `data_generator.py`: aumentar la separabilidad entre
      clases en los datos sintéticos (features más correlacionadas con
      el target de forma realista)
- [ ] Alternativamente: avanzar directamente a datos reales de los
      2 distritos disponibles (Miraflores + Magdalena del Mar)
- [ ] Con datos reales: esperar AUC > 0.70 como mínimo aceptable

---

## Conclusión para el paper

> "La validación inicial del pipeline se realizó con un dataset sintético
> de N muestras distribuidas en 10 distritos simulados. El AUC-ROC de 0.54
> obtenido confirma que el modelo no extrae señal predictiva de datos
> sintéticos sin estructura espacial real, resultado consistente con la
> expectativa teórica. La evaluación definitiva del modelo se realizará
> con datos georreferenciados reales de Lima Metropolitana."