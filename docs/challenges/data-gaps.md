# Desafíos: Gaps de datos y estrategias de mitigación

---

## Gap 1: INEI sin granularidad de manzana en formato procesable

**Descripción:**
Los datos censales de INEI (Censo 2017) con nivel de manzana no están
disponibles públicamente en formato directamente procesable (CSV/API).
Los productos descargables ofrecen granularidad distrital o de zona censal.

**Impacto:**
Las features de densidad poblacional son menos granulares de lo ideal.
Una celda de 500m × 500m puede cruzar múltiples zonas censales con
características heterogéneas.

**Estrategia aplicada:**
- Usar datos a nivel de zona censal (mayor granularidad disponible)
- Interpolar con límites de manzana obtenidos de OSM cuando sea necesario
- Documentar como limitación en la sección de trabajo futuro del paper

---

## Gap 2: Dataset EMI simulado, no datos reales de generación

**Descripción:**
No se dispone de datos reales de la Empresa Municipal de Ingeniería (EMI)
sobre volumen y frecuencia de recolección por zona. El dataset actual
(`data/simulated/`) fue generado sintéticamente para desarrollo.

**Impacto:**
Las features derivadas de generación de residuos tienen sesgo de simulación.
La distribución espacial sintética no refleja los patrones reales de Lima.

**Estrategia aplicada:**
- Usar datos MINAM (SIGERSOL) a nivel distrital como proxy
- El pipeline está diseñado para ser agnóstico a la fuente: reemplazar
  el CSV simulado por datos reales no requiere cambios en el código
- Migración a datos reales planificada conforme avance TP1

---

## Gap 3: Cobertura de puntos de reciclaje limitada a 2 distritos

**Descripción:**
Los puntos de reciclaje recolectados manualmente cubren solo Miraflores
(31 puntos) y Magdalena del Mar (22 puntos). Total: 53 puntos georreferenciados.

**Impacto:**
- El modelo entrenado solo con 2 distritos no puede reclamar generalización
  a los 50 distritos de Lima Metropolitana
- El spatial CV no es aplicable con solo 2 distritos
- Sesgo geográfico: Miraflores y Magdalena son distritos de NSE alto-medio,
  subrepresentando zonas periféricas

**Estrategia aplicada:**
- Expansión progresiva en curso: recolección manual vía Google Maps +
  fuentes municipales de distritos adicionales
- Declarar explícitamente en el paper que el alcance actual es una
  prueba de concepto del pipeline, no un modelo Lima-wide validado
- Priorizar distritos de NSE variado para reducir sesgo en entrenamiento

---

## Gap 4: Ausencia de datos de participación ciudadana por zona

**Descripción:**
No existe una fuente pública con tasas de uso o participación por punto
de reciclaje en Lima. No se puede medir la demanda real, solo la oferta.

**Impacto:**
El modelo aprende de dónde existen puntos (proxy de dónde funcionan),
no de dónde la ciudadanía tiene mayor propensión a reciclar.

**Estrategia aplicada:**
- Proxy: densidad poblacional + NSE como indicadores correlacionados
  con participación en reciclaje (justificado por J-PAL, 2013)
- Documentar como limitación estructural del dataset

---

## Gap 5: Datos OSM con cobertura irregular en Lima periférica

**Descripción:**
OpenStreetMap tiene buena cobertura en distritos centrales (Miraflores,
San Isidro, Barranco) pero cobertura reducida en distritos periféricos
(Carabayllo, Puente Piedra, Lurigancho-Chosica).

**Impacto:**
Las features derivadas de OSM (accesibilidad peatonal, densidad de POIs,
red vial) serán menos confiables en celdas de distritos periféricos.

**Estrategia aplicada:**
- Cuantificar cobertura OSM por distrito antes del entrenamiento
- Excluir distritos con cobertura < umbral definido del training set
- Reportar el Area of Applicability (Meyer & Pebesma, 2021) del modelo
  como métrica de confiabilidad geográfica