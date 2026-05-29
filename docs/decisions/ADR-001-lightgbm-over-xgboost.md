# ADR-001: LightGBM sobre XGBoost

**Fecha:** 2026-04  
**Estado:** Aceptado  
**Decisores:** Nikole García Chávez, asesor Peter Jonathan Montalvo Garcia

---

## Contexto

Durante el TDP (ciclo anterior) se realizó un benchmarking comparando
Random Forest, XGBoost, Gradient Boosting Regression y LightGBM en criterios
de precisión, interpretabilidad, escalabilidad, robustez y generalización.
XGBoost resultó ganador del benchmarking TDP.

Al iniciar TP1 (2026-10) se re-evaluó la decisión considerando:
- El perfil real del dataset: ~11,000 celdas espaciales de 500m × 500m
- Features tabulares derivadas de geometrías OSM (alta cardinalidad)
- RAM limitada en instancias Azure for Students
- Necesidad de explicaciones por instancia en tiempo de inferencia (SHAP)

---

## Opciones evaluadas

| Criterio | XGBoost | LightGBM |
|---|---|---|
| Estrategia de crecimiento | Level-wise | Leaf-wise |
| Uso de memoria | Mayor | Menor |
| Velocidad de entrenamiento | Más lento en datasets grandes | Más rápido |
| SHAP TreeExplainer | Compatible | Más eficiente nativamente |
| Soporte features categóricas | Requiere encoding manual | Nativo |
| Documentación geoespacial | Mayor | Menor |

---

## Decisión

**LightGBM** confirmado por el asesor Peter Montalvo en TP1 como
algoritmo definitivo del proyecto.

---

## Consecuencias

**Positivas:**
- Menor footprint de memoria — viable en instancias Azure gratuitas
- TreeExplainer de SHAP más eficiente computacionalmente con LightGBM
- Entrenamiento más rápido permite más iteraciones de hiperparámetros

**Negativas / Riesgos:**
- LightGBM puede sobreajustar en datasets pequeños → mitigado con
  `min_data_in_leaf` y regularización (`lambda_l1`, `lambda_l2`)
- Menos casos de uso documentados en literatura de ML geoespacial
  → mitigado con spatial CV siguiendo Valavi et al. (2019)

**Pendiente:**
- Validar rendimiento comparativo cuando se expanda a más distritos