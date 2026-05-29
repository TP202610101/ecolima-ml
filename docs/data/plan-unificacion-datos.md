# Plan de unificación de datos de municipalidades

---

## Problema central

Cada municipalidad entregará los datos en un formato diferente:
- Distintos nombres de columnas ("Latitud" vs "lat" vs "Y")
- Distintas convenciones de coordenadas (decimal vs grados-minutos-segundos)
- Distintos vocabularios para tipos de residuo ("PET" vs "Plástico" vs "Botellas")
- Distintos estados de completitud (con o sin coordenadas, con o sin capacidad)

El plan de unificación convierte todo eso al schema definido en PostGIS.

---

## Flujo de recepción y unificación