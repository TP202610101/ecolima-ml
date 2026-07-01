# Changelog

Registro cronológico de decisiones, cambios y avances del pipeline ML.
Formato: [YYYY-MM-DD] — descripción.

---

## [2026-05-21]
- Creación del repositorio ecolima-ml
- Migración de archivos existentes a estructura documentada
- Primer README con estado actual del pipeline

---

## [Por completar conforme avances]

<!-- Ejemplo de entradas que irás agregando: -->

<!--
## [2026-XX-XX]
- Implementado spatial cross-validation por distrito
- Decisión: umbral de clasificación binaria en 0.5 (ver ADR-002)

## [2026-XX-XX]
- Expansión de datos reales: agregados puntos de reciclaje de San Isidro
- Reentrenamiento del modelo con 3 distritos — F1 subió de X a Y
-->

## [2026-07-01]
- Preparado dataset real provisional `data/real/dataset_real_candidatos_v0_3.csv` sin adoptar etiquetas heredadas como target final.
- Agregado flujo de entrenamiento con data simulada `scripts/run_simulated_training.py`.
- Agregada capa de servicio y API FastAPI opcional para contratos de endpoints ML.
- Separado uso tecnico de dataset simulado frente a evidencia metodologica real.
