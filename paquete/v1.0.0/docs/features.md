<!-- Fecha de creación: 2026-10-03 -->
<!-- Fecha de modificación: 2026-10-03 -->

# Variables explicativas del dataset v1.0.0

Dataset `dataset_v1.0.0.csv` (sha256 `aa6f277925473681eb9d1012ff0eb11065d0f1a7b3d20acfcacfb598e2568154`): 2242 sitios, 47 puntos de reciclaje existentes (positivos) y 2195 sitios candidatos de fondo no etiquetado. Coordenadas EPSG:4326. Las 15 variables se calcularon desde los archivos fuente listados en `fuentes.csv` (mismo sha256) y se entregan congeladas: este paquete no las recalcula.

Reglas comunes:

- Ninguna variable se deriva del inventario de puntos de reciclaje (no hay distancias ni densidades de puntos existentes): evita que el modelo aprenda la etiqueta a partir de sí misma.
- No se usa la composición de residuos de SIGERSOL.
- Radios de 250, 500 y 1 000 m alrededor del sitio; vías y POIs se miden en UTM 18S (EPSG:32718).
- Transformación en el modelo: log1p a población, vías y POIs; estandarización de todas las numéricas; one-hot de `clase_oportunidad`.

| # | variable | unidad | radio | fuente | definición | en el modelo |
|---|---|---|---|---|---|---|
| 1 | `clase_oportunidad` | categoría | — | Universo de candidatos (OSM) | Clase del sitio de oportunidad más cercano del universo de candidatos (para los sitios de fondo, la propia): paradero de transporte público, mercado municipal (no confirmado), centro comercial o parque. | one-hot |
| 2 | `pob_wp_r250` | personas | 250 m | WorldPop 2020 (1 km) | Suma de población por píxel × fracción del píxel dentro del círculo de radio 250 m (10×10 subpuntos por píxel, distancia equirectangular local). Con píxeles de ~1 km, a 250 m es una fracción de un solo píxel. | log1p + estandarización |
| 3 | `pob_wp_r500` | personas | 500 m | WorldPop 2020 (1 km) | Suma de población por píxel × fracción del píxel dentro del círculo de radio 500 m (10×10 subpuntos por píxel, distancia equirectangular local). | log1p + estandarización |
| 4 | `pob_wp_r1000` | personas | 1000 m | WorldPop 2020 (1 km) | Suma de población por píxel × fracción del píxel dentro del círculo de radio 1000 m (10×10 subpuntos por píxel, distancia equirectangular local). | log1p + estandarización |
| 5 | `vias_km_km2_r250` | km/km² | 250 m | OpenStreetMap | Longitud de vías OSM (highway: motorway, trunk, primary, secondary, tertiary y sus enlaces, unclassified, residential, living_street, service, pedestrian) recortada al círculo de radio 250 m en UTM 18S (EPSG:32718), dividida por el área del círculo. | log1p + estandarización |
| 6 | `vias_km_km2_r500` | km/km² | 500 m | OpenStreetMap | Longitud de vías OSM (highway: motorway, trunk, primary, secondary, tertiary y sus enlaces, unclassified, residential, living_street, service, pedestrian) recortada al círculo de radio 500 m en UTM 18S (EPSG:32718), dividida por el área del círculo. | log1p + estandarización |
| 7 | `vias_km_km2_r1000` | km/km² | 1000 m | OpenStreetMap | Longitud de vías OSM (highway: motorway, trunk, primary, secondary, tertiary y sus enlaces, unclassified, residential, living_street, service, pedestrian) recortada al círculo de radio 1000 m en UTM 18S (EPSG:32718), dividida por el área del círculo. | log1p + estandarización |
| 8 | `poi_n_km2_r250` | POI/km² | 250 m | OpenStreetMap | Nodos OSM con amenity, shop, leisure, tourism, office, craft o healthcare a ≤250 m (UTM 18S), sin residuos/reciclaje (recycling, waste_basket, waste_disposal, waste_transfer_station) ni el nodo propio, divididos por πr². | log1p + estandarización |
| 9 | `poi_n_km2_r500` | POI/km² | 500 m | OpenStreetMap | Nodos OSM con amenity, shop, leisure, tourism, office, craft o healthcare a ≤500 m (UTM 18S), sin residuos/reciclaje (recycling, waste_basket, waste_disposal, waste_transfer_station) ni el nodo propio, divididos por πr². | log1p + estandarización |
| 10 | `poi_n_km2_r1000` | POI/km² | 1000 m | OpenStreetMap | Nodos OSM con amenity, shop, leisure, tourism, office, craft o healthcare a ≤1000 m (UTM 18S), sin residuos/reciclaje (recycling, waste_basket, waste_disposal, waste_transfer_station) ni el nodo propio, divididos por πr². | log1p + estandarización |
| 11 | `ghsl_frac_construida_r250` | proporción 0–1 | 250 m | GHS-BUILT-S 2020 (100 m) | Suma de m² construidos de las celdas cuyo centro está a ≤250 m, dividida por (n celdas × 10 000 m²); nodata = 0. | estandarización |
| 12 | `ghsl_frac_construida_r500` | proporción 0–1 | 500 m | GHS-BUILT-S 2020 (100 m) | Suma de m² construidos de las celdas cuyo centro está a ≤500 m, dividida por (n celdas × 10 000 m²); nodata = 0. | estandarización |
| 13 | `ghsl_frac_construida_r1000` | proporción 0–1 | 1000 m | GHS-BUILT-S 2020 (100 m) | Suma de m² construidos de las celdas cuyo centro está a ≤1000 m, dividida por (n celdas × 10 000 m²); nodata = 0. | estandarización |
| 14 | `pendiente_grados` | grados | píxel | Copernicus DEM GLO-30 | Pendiente de Horn (ventana 3×3) sobre el mosaico de 4 teselas, en el píxel del sitio. | estandarización |
| 15 | `sigersol_gpc_municipal_kg_hab_dia` | kg/hab/día | distrito | MINAM SIGERSOL + límites IDEP | Generación per cápita municipal (GENERACION_PER_CAPITA_MUNICIPAL) del distrito del sitio (UBIGEO por IDEP), último año con valor > 0 entre 2019 y 2024. Mismo valor para todos los sitios de un distrito. | estandarización |

## Distribución observada (2 242 sitios)

| variable | mínimo | P25 | mediana | P75 | máximo |
|---|---|---|---|---|---|
| `pob_wp_r250` | 85.9 | 2025 | 2966 | 4101 | 6536 |
| `pob_wp_r500` | 349.7 | 8051 | 1.197e+04 | 1.638e+04 | 2.405e+04 |
| `pob_wp_r1000` | 1338 | 3.202e+04 | 4.751e+04 | 6.258e+04 | 9.325e+04 |
| `vias_km_km2_r250` | 6.231 | 23.35 | 27.49 | 31.49 | 47.35 |
| `vias_km_km2_r500` | 6.745 | 22.03 | 25.91 | 28.71 | 38.86 |
| `vias_km_km2_r1000` | 4.726 | 21.1 | 24.69 | 27.01 | 32.86 |
| `poi_n_km2_r250` | 0 | 35.65 | 86.58 | 193.5 | 2343 |
| `poi_n_km2_r500` | 0 | 39.47 | 77.67 | 157.6 | 1168 |
| `poi_n_km2_r1000` | 0 | 40.43 | 74.17 | 132.3 | 510.6 |
| `ghsl_frac_construida_r250` | 0.0005222 | 0.3578 | 0.4049 | 0.4433 | 0.6338 |
| `ghsl_frac_construida_r500` | 0.005448 | 0.3699 | 0.4132 | 0.444 | 0.5298 |
| `ghsl_frac_construida_r1000` | 0.01932 | 0.3427 | 0.4028 | 0.4296 | 0.5058 |
| `pendiente_grados` | 0.06216 | 1.346 | 2.2 | 3.604 | 31.93 |
| `sigersol_gpc_municipal_kg_hab_dia` | 0.64 | 0.91 | 0.97 | 1.09 | 1.14 |

`clase_oportunidad`: paradero_transporte_publico (2079), mercado_municipal_no_confirmado (79), centro_comercial (71), parque (13).

## Límites

- WorldPop tiene píxeles de ~1 km: a 250 m la población es una fracción de un píxel, no un conteo local.
- La densidad vial es un proxy de accesibilidad, no tiempo de caminata sobre la red peatonal.
- OSM tiene cobertura desigual entre distritos; los POIs y las vías heredan ese sesgo de mapeo.
- La generación per cápita es distrital: no distingue entre sitios del mismo distrito.
