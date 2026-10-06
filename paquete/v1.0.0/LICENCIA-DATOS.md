<!-- Fecha de creación: 2026-10-03 -->
<!-- Fecha de modificación: 2026-10-03 -->

# Licencia de los datos — paquete EcoLima v1.0.0

Los datos de este paquete forman una base derivada de OpenStreetMap y se ponen a disposición bajo la Open
Database License 1.0 (https://opendatacommons.org/licenses/odbl/1-0/). Contienen además información de
WorldPop y GHSL (CC BY 4.0), Copernicus DEM, MINAM-SIGERSOL e IGN (ODC-By) y de la Municipalidad
Distrital de San Isidro; los avisos completos de cada fuente están en `datos/fuentes.csv`.

Texto de la licencia: https://opendatacommons.org/licenses/odbl/1-0/

## Qué cubre

- **Datos** (`datos/` y `salidas/`, incluidos `datos/distritos.geojson`): base derivada de OpenStreetMap, bajo ODbL 1.0.
  Cada fuente conserva su atribución (tabla de abajo y `datos/fuentes.csv`).
- **Parámetros del modelo** (`modelo/`): obra producida a partir de esa base; se usan con el aviso de pie de mapa.
- **Mapas y reportes** elaborados con estos datos: obras producidas; deben mostrar el aviso siguiente y enlazar a la
  página de fuentes y licencias.
- **Código** que genera o consume estos datos: no forma parte de la base de datos; se licencia aparte.

Aviso para el pie de un mapa o reporte:

> Datos: © colaboradores de OpenStreetMap (ODbL) · WorldPop (CC BY 4.0) · GHSL, JRC (CC BY 4.0) · Copernicus DEM · MINAM-SIGERSOL e IGN/INEI (ODC-By) · Municipalidad de San Isidro. [Fuentes y licencias]

## Fuentes y avisos

| código | licencia | atribución |
|---|---|---|
| `COPDEM-S12-W077` | Licencia Copernicus DEM GLO-30 (uso libre con atribución) | produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30 |
| `COPDEM-S12-W078` | Licencia Copernicus DEM GLO-30 (uso libre con atribución) | produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30 |
| `COPDEM-S13-W077` | Licencia Copernicus DEM GLO-30 (uso libre con atribución) | produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30 |
| `COPDEM-S13-W078` | Licencia Copernicus DEM GLO-30 (uso libre con atribución) | produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30 |
| `GHSL-BUILT-2020` | Creative Commons Attribution 4.0 International (CC BY 4.0) | Pesaresi, M.; Politis, P. (2023). GHS-BUILT-S R2023A - GHS built-up surface grid. European Commission, Joint Research Centre. doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA |
| `IDEP-LIM-LIMA` | Open Data Commons Attribution License (ODC-By), declarada por el IGN para la misma capa en la Plataforma Nacional de Datos Abiertos | Instituto Geográfico Nacional (IGN), Límites Distritales, Plataforma Nacional de Datos Abiertos (datosabiertos.gob.pe), ODC-By; límites de origen INEI |
| `LOTE-SAN-ISIDRO-V0.2` | Sin licencia explícita. Documento oficial municipal (Resolución de Alcaldía N.° 2026-017-ALC/MSI); los textos oficiales y los simples datos no están protegidos por derecho de autor (D. Leg. 822, art. 9 b y d); se cita la fuente | Municipalidad Distrital de San Isidro, Plan Anual de Trabajo 2026 «Recicla San Isidro», aprobado por Resolución de Alcaldía N.° 2026-017-ALC/MSI, publicado en gob.pe el 2026-01-19 |
| `MINAM-GEN-2019-2024` | Open Data Commons Attribution License (ODC-By) | Ministerio del Ambiente del Perú - SIGERSOL, Plataforma Nacional de Datos Abiertos |
| `OSM-PER-20260918` | Open Data Commons Open Database License (ODbL) 1.0 | © OpenStreetMap contributors, disponible bajo ODbL 1.0 |
| `UNIVERSO-CANDIDATOS-V0.1` | Open Data Commons Open Database License (ODbL) 1.0 (derivado de OSM) | © OpenStreetMap contributors, disponible bajo ODbL 1.0 |
| `WORLDPOP-PER-2020` | Creative Commons Attribution 4.0 International (CC BY 4.0) | WorldPop (www.worldpop.org) y CIESIN, Columbia University (2018). Global High Resolution Population Denominators Project. doi:10.5258/SOTON/WP00670 |
