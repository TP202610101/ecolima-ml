# ADR-002: Clasificación binaria vs. regresión de idoneidad

**Fecha:** 2026-04  
**Estado:** Pendiente confirmación asesor  
**Decisores:** Nikole García Chávez

---

## Contexto

El modelo puede formularse de dos maneras:

1. **Clasificación binaria:** `is_suitable` ∈ {0, 1} — la celda es óptima
   o no lo es. La etiqueta positiva se deriva por proximidad a puntos de
   reciclaje reales existentes (umbral: 250m).

2. **Regresión de idoneidad:** `suitability_score` ∈ [0, 1] — score continuo
   que refleja qué tan óptima es la celda. Más informativo para ranking,
   pero requiere definir la variable target de forma continua.

---

## Opciones evaluadas

| Criterio | Clasificación binaria | Regresión |
|---|---|---|
| Definición del target | Clara (proximidad a punto real) | Requiere función de idoneidad compuesta |
| Métricas | F1, AUC-ROC, Precision, Recall | R², RMSE, MAE |
| Desbalance de clases | Problema explícito → SMOTE o class_weight | No aplica directamente |
| Interpretabilidad SHAP | SHAP values por clase | SHAP values sobre score continuo |
| Utilidad para ranking | Requiere usar probabilidades (.predict_proba) | Directo |

---

## Decisión

**Pendiente.** Se presentarán ambas opciones al asesor en la próxima reunión.

Postura preliminar: clasificación binaria con `.predict_proba()` para ranking,
dado que la disponibilidad de etiquetas reales favorece la formulación discreta
(punto de reciclaje existe / no existe en radio de proximidad).

---

## Consecuencias (si se elige clasificación)

- Se necesita definir el umbral de proximidad para generar `is_suitable`
  (propuesta: 250m basado en accesibilidad peatonal de 3 min)
- El dataset será desbalanceado: pocas celdas positivas vs miles de negativas
  → evaluar `scale_pos_weight` en LightGBM o submuestreo