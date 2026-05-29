# ADR-003: Estrategia de validación cruzada espacial

**Fecha:** 2026-04  
**Estado:** Definido, implementación pendiente  
**Decisores:** Nikole García Chávez

---

## Contexto

El split aleatorio estándar (train/test random) produce data leakage espacial
en datos georreferenciados: celdas de entrenamiento y prueba geográficamente
adyacentes comparten autocorrelación espacial, inflando artificialmente las
métricas.

---

## Decisión

**Spatial cross-validation por distrito:**
- Las celdas se asignan a folds por distrito, no aleatoriamente
- En cada fold, un distrito completo queda fuera del entrenamiento (held-out)
- Evalúa la capacidad de generalización a distritos no vistos

**Fundamento académico:**
- Valavi et al. (2019) — blockCV: spatial block cross-validation
- Koldasbayeva et al. (2024) — justifica spatial CV en modelos geoespaciales

---

## Consecuencias

**Positivas:**
- Estimación honesta del rendimiento en distritos no representados
- Justificable metodológicamente ante revisores académicos
- Permite cuantificar el Area of Applicability (Meyer & Pebesma, 2021)

**Negativas / Limitaciones:**
- Con solo 2 distritos actuales (Miraflores, Magdalena del Mar), el spatial CV
  no es aplicable todavía — solo se puede hacer split simple por distrito
- Se requieren al menos 5–6 distritos para k-fold espacial con k=5
  → expansión de datos es prerequisito para esta etapa

**Implementación:**
- Corto plazo: split simple, distrito Miraflores = train, Magdalena = test
- Mediano plazo: implementar spatial k-fold cuando haya ≥5 distritos