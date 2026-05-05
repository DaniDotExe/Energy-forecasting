# Proyecto de Pronóstico de Energía Solar - Planta "El Paso"

Este proyecto contiene una serie de scripts en Python diseñados para la extracción, procesamiento y consolidación de datos climáticos y de generación eléctrica para la planta solar "El Paso" (Cesar, Colombia). El objetivo final es generar datasets limpios para realizar predicciones de la generación de energía solar con diferentes modelos de aprendizaje automático.

Ubicación de la planta: 9.785041627157241, -73.7201783147516
Rango de fecha de los datos: 2019-01-01 a 2023-12-31

## Estructura del Proyecto
El repositorio está organizado de la siguiente manera para mantener la trazabilidad y el orden en el flujo de datos:

- `src/`: Contiene los scripts de Python para la extracción de APIs (`API_OpenMeteo_Solar.py`, `API_NASAPOWER.py`) y futuros scripts de modelado.
- `data/`: Almacena los archivos CSV generados por las APIs y los datos históricos de XM.
- `output/`: Directorio destinado a guardar los resultados de entrenamiento de los modelos (pesos, logs, métricas).
- `graficas/`: Carpeta para visualizaciones de análisis exploratorio (EDA) y resultados de predicción.

## Metodología de Procesamiento
Para asegurar la consistencia entre las fuentes climáticas y los datos de generación real, se aplica el siguiente criterio de procesamiento:

1.  **Filtro de Horas Diurnas**: Solo se conservan los registros entre las **06:00 AM y las 05:00 PM**. Esto optimiza el entrenamiento de los modelos al eliminar períodos de generación nula (noche) que podrían sesgar los resultados.
2.  **Alineación Temporal**: Todos los datos se normalizan a la zona horaria local de la planta para permitir la fusión (merge) precisa de los datasets.
3.  **Variables Estandarizadas**: Independientemente de la fuente, los datos se mapean a nombres de columnas uniformes (`Irradiancia_It`, `Temperatura_Tt`, etc.).

## Selección de Variables Climáticas
Se han seleccionado cuatro variables fundamentales para predecir la generación de energía solar. Esta elección se basa en su impacto directo sobre el rendimiento de las celdas fotovoltaicas:

1.  **Irradiancia (`Irradiancia_It`)**: Es el factor principal. Representa la potencia de la radiación solar por unidad de área, que se convierte directamente en electricidad.
2.  **Temperatura (`Temperatura_Tt`)**: Las celdas solares pierden eficiencia a medida que aumenta su temperatura de operación.
3.  **Humedad (`Humedad_Ht`)**: Afecta la dispersión de la luz y la refrigeración por evaporación de los paneles.
4.  **Viento (`Viento_Wt`)**: Actúa como un agente refrigerante natural, ayudando a mantener los paneles en temperaturas de operación más eficientes.

## Fuentes de Datos Definitivas
Para garantizar la robustez del modelo, se utilizan tres fuentes principales de información: dos para variables climáticas y una para la generación real.

### 1. Open-Meteo (Modelo ERA5 / ERA5-Land)
[Open-Meteo](https://open-meteo.com/) utiliza el modelo **ERA5** del Centro Europeo de Previsiones Meteorológicas a Plazo Medio (ECMWF).
-   **Origen:** Combina observaciones de satélites, estaciones terrestres, boyas y aviones con modelos climáticos globales (Asimilación de datos).
-   **Nivel de Confianza:** **Muy Alto**. Es el estándar de oro en la comunidad científica para el reanálisis climático global.
-   **Archivo de salida:** `data/openmeteo-horario.csv`

| Variable | Columna CSV | Unidad | Descripción |
| :--- | :--- | :--- | :--- |
| **Fecha y Hora** | `Fecha_Hora` | YYYY-MM-DD HH:MM | Marca de tiempo en zona horaria local. |
| **Irradiancia** | `Irradiancia_It` | W/m² | Irradiancia de onda corta global. |
| **Temperatura** | `Temperatura_Tt` | °C | Temperatura del aire a 2 metros de altura. |
| **Humedad** | `Humedad_Ht` | % | Humedad relativa a 2 metros de altura. |
| **Viento** | `Viento_Wt` | km/h | Velocidad del viento a 10 metros de altura. |

### 2. NASA POWER (Datos Satelitales)
[NASA POWER](https://power.larc.nasa.gov/) (Prediction Of Worldwide Energy Resources) proporciona datos derivados de la misión satelital GEOS (Goddard Earth Observing System) y modelos de asimilación de datos atmosféricos.
-   **Origen:** Datos obtenidos principalmente mediante sensores satelitales calibrados para estudios de energía renovable y agricultura.
-   **Nivel de Confianza:** **Alto**. Muy utilizado en la industria solar para estimar el recurso disponible en sitios sin estaciones meteorológicas cercanas.
-   **Archivo de salida:** `data/nasapower-horario.csv`

| Variable API | Columna CSV | Unidad | Descripción |
| :--- | :--- | :--- | :--- |
| `time_local` | `Fecha_Hora` | YYYY-MM-DD HH:MM | Marca de tiempo en zona horaria local. |
| `ALLSKY_SFC_SW_DWN` | `Irradiancia_It` | Wh/m² | Irradiancia de onda corta hacia abajo (superficie). |
| `T2M` | `Temperatura_Tt` | °C | Temperatura a 2 metros. |
| `RH2M` | `Humedad_Ht` | % | Humedad relativa a 2 metros. |
| `WS10M` | `Viento_Wt` | km/h | Velocidad del viento a 10 metros (convertido de m/s). |

### 3. XM (Generación Real)
[XM](https://www.xm.com.co/) es el operador del Sistema Interconectado Nacional (SIN) y administrador del Mercado de Energía Mayorista (MEM) en Colombia.
-   **Origen:** Sensores de medición directa (telemedida) instalados en la planta solar "El Paso". Son datos oficiales de despacho y generación.
-   **Nivel de Confianza:** **Máximo (Referencia)**. Representa la verdad de campo (ground truth) contra la cual se entrenan y validan los modelos de pronóstico.
-   **Archivo consolidado:** `data/data_2019_2023_kwH_xm.csv`

| Variable | Columna CSV | Unidad | Descripción |
| :--- | :--- | :--- | :--- |
| **Fecha** | `Fecha` | YYYY-MM-DD | Fecha del reporte de generación. |
| **Recurso** | `Recurso` | Texto | Nombre de la planta (EL PASO). |
| **Generación Horaria** | `0, 1, ..., 23` | kWh | Energía real generada en cada intervalo de una hora. |
