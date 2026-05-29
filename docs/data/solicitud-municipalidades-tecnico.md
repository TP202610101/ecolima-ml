# Especificación técnica de datos requeridos — Municipalidades de Lima

**Proyecto:** EcoLima ML — Recomendación de puntos de reciclaje con ML
**Institución:** Universidad Peruana de Ciencias Aplicadas (UPC)
**Contacto:** Nikole García Chávez / Alexander Cantoral Sedamano

---

## Municipalidades priorizadas

### Tier 1 — Datos reales ya parcialmente recolectados
| Municipalidad | Distritos | Estado |
|---|---|---|
| Municipalidad de Miraflores | Miraflores | ✅ 31 puntos recolectados manualmente |
| Municipalidad de Magdalena del Mar | Magdalena del Mar | ✅ 22 puntos recolectados manualmente |

### Tier 2 — Solicitud prioritaria (variedad de NSE)
| Municipalidad | Distritos | Justificación |
|---|---|---|
| Municipalidad de San Juan de Lurigancho | SJL | Distrito más poblado de Lima (~1.1M hab) |
| Municipalidad de Villa El Salvador | VES | NSE C/D, contraste con Miraflores |
| Municipalidad de San Martín de Porres | SMP | NSE C, zona norte, alta densidad |
| Municipalidad de Surco | Santiago de Surco | NSE A/B, referencia comparativa |
| Municipalidad de Comas | Comas | NSE C/D, zona norte periférica |

### Tier 3 — Solicitud secundaria
Ate, Los Olivos, Callao (si está disponible), La Molina, San Borja.

---

## Dataset 1: Puntos de reciclaje existentes

**Tabla destino:** `recycling_points`

| Campo | Tipo | Descripción | Obligatorio |
|---|---|---|---|
| Nombre del punto | texto | Nombre o identificador del contenedor/punto | ✅ |
| Dirección | texto | Dirección completa (calle + número + referencia) | ✅ |
| Latitud | decimal (7 dígitos) | Coordenada WGS84, ej: -12.1211234 | ✅ |
| Longitud | decimal (7 dígitos) | Coordenada WGS84, ej: -77.0234567 | ✅ |
| Tipo de residuo aceptado | texto/lista | Papel, plástico, vidrio, metales, orgánicos, RAEE | ✅ |
| Estado operativo | booleano/texto | Activo / Inactivo / En mantenimiento | ✅ |
| Capacidad del contenedor | número (litros) | Volumen total del contenedor | Deseable |
| Frecuencia de vaciado | texto | Diaria / semanal / quincenal | Deseable |
| Fecha de instalación | fecha (YYYY-MM-DD) | Cuándo se instaló el punto | Deseable |
| Responsable de mantenimiento | texto | Área o empresa a cargo | Opcional |

**Volumen esperado por distrito:**
- Mejor caso: 50–200 puntos con coordenadas validadas y todos los campos
- Peor caso: 10–30 puntos, solo nombre + dirección (sin coordenadas)
- Si no hay coordenadas: se georreferencian manualmente vía Google Maps
  (proceso: ~2 min/punto → 300 puntos ≈ 10 horas de trabajo)

**Formatos aceptados (en orden de preferencia):**
1. CSV o Excel con columnas lat/lon separadas
2. GeoJSON o Shapefile (exportado desde SIG municipal)
3. PDF con listado de direcciones (último recurso — requiere geocodificación manual)
4. Respuesta por correo con tabla en el cuerpo del mensaje

---

## Dataset 2: Generación de residuos por distrito

**Tabla destino:** `waste_generation`

| Campo | Tipo | Descripción | Obligatorio |
|---|---|---|---|
| Año | entero | Año del dato, ej: 2023 | ✅ |
| Distrito | texto | Nombre del distrito según UBIGEO INEI | ✅ |
| Toneladas/día generadas | decimal | Total de residuos sólidos generados por día | ✅ |
| Kg/habitante/día | decimal | Generación per cápita | Deseable |
| % residuos reciclables | decimal (0–100) | Fracción valorizable del total | Deseable |
| Toneladas recicladas/mes | decimal | Cantidad efectivamente reciclada | Deseable |
| Fuente del dato | texto | EMI, MINAM, SIGERSOL, estudio propio | ✅ |

**Volumen esperado:**
- Mejor caso: serie histórica 2018–2024 con granularidad mensual por distrito
- Peor caso: un único dato anual 2022 o 2023 a nivel distrital
- Fuente alternativa si la municipalidad no tiene: SIGERSOL/MINAM
  (disponible públicamente pero con menor granularidad)

**Formatos aceptados:**
1. Excel o CSV
2. Informe de gestión ambiental en PDF (se extraen tablas manualmente)
3. Datos del SIGERSOL ya consolidados por el equipo

---

## Dataset 3: Rutas de recolección (opcional pero valioso)

**Tabla destino:** referencia para `candidate_zones`

| Campo | Tipo | Descripción |
|---|---|---|
| ID de ruta | texto | Identificador de la ruta |
| Frecuencia | texto | Diaria, interdiaria, semanal |
| Zonas cubiertas | polígono o lista de calles | Área de cobertura |
| Horario | texto | Turno de recolección |

**Nota:** Este dataset es secundario. Si no está disponible, se omite
sin afectar el modelo principal.

---

## Estimación de impacto por volumen de datos

| Escenario | Distritos | Puntos reciclaje | AUC esperado | Spatial CV |
|---|---|---|---|---|
| Mínimo viable | 2 (actual) | 53 | ~0.65–0.70 | No aplicable |
| Aceptable | 5–6 | 150–300 | ~0.72–0.78 | k=5 posible |
| Bueno | 10+ | 500+ | ~0.78–0.85 | Robusto |
| Ideal | 20+ | 1000+ | > 0.85 | Lima-wide |