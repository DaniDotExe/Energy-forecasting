# Proyecto de Pronóstico de Energía Solar - Planta "El Paso"

Este proyecto contiene una serie de scripts en Python diseñados para la extracción, procesamiento y consolidación de datos climáticos y de generación eléctrica para la planta solar "El Paso" (Cesar, Colombia). El objetivo final es generar datasets limpios para realizar predicciones de la generación de energía solar con diferentes modelos de aprendizaje automático.

Ubicación de la planta: 9.785041627157241, -73.7201783147516
Rango de fecha de los datos: 2019-01-01 a 2023-12-31

## Estructura del Proyecto
El repositorio está organizado de la siguiente manera:

- `data/`: Almacena los archivos CSV originales de las APIs, los datos históricos de XM y el dataset consolidado (`data-hourly.csv`).
- `src/`: Scripts de Python para procesamiento y análisis:
    - `API_NASAPOWER.py` / `API_OpenMeteo_Solar.py`: Extracción de datos climáticos históricos.
    - `dataset_maker.py`: Genera el dataset final (`data-hourly.csv`) uniendo clima y generación en el rango 06:00-17:00.
    - `Analyze_xm.py`: EDA de los datos de generación de XM (estacionalidad, perfiles diarios).
    - `Analyze_NASA_xm_correlation.py`: Análisis de correlación clima-generación y dependencia temporal (ACF/PACF).
- `EDA/`: Visualizaciones generadas por los scripts de análisis:
    - `XM/`: Análisis detallado de la planta solar (perfiles por hora, promedios diarios, etc.).
    - `NASA_xm_correlation/`: Heatmaps de correlación y gráficos de dispersión.
    - `FINAL_COMPARISON/`: Resultados de la comparativa final horaria (SARIMAX vs LSTM).
    - `MONTHLY_COMPARISON/`: Resultados de la batalla final mensual (SARIMAX vs LSTM vs MLP).
    - `LSTM_grid_search/`: Comparativa de rendimiento según ventana de memoria (7, 14, 30 días).
- `output/`: Directorio para modelos entrenados y resultados técnicos.

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
-   **Justificación de Selección:** Se seleccionó esta fuente como base principal para el dataset consolidado debido a que **no presenta valores faltantes (nulls)** en el rango de tiempo estudiado.
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
| **Versión** | `Version` | Texto | Código de versión del dato reportado. |

### 4. Dataset Consolidado (`data-hourly.csv`)
Este archivo es el resultado de ejecutar `dataset_maker.py`. Combina las variables de NASA POWER con la generación real de XM.
- **Rango:** 06:00 AM a 05:00 PM (12 registros por día).
- **Contenido:** `Fecha_Hora`, `Irradiancia_It`, `Temperatura_Tt`, `Humedad_Ht`, `Viento_Wt`, `kWh`.

## Fase de Modelado Avanzado

En esta fase se implementaron modelos de aprendizaje profundo (Deep Learning) y estadísticos para predecir la generación eléctrica.

### 1. Modelos Horarios (High-Complexity LSTM)
Se diseñó un modelo **LSTM Bidireccional** de alta complejidad con las siguientes características:
- **Arquitectura:** 3 capas de LSTM Bidireccional, 128 neuronas por capa, con `LayerNormalization` y `Dropout(0.3)`.
- **Ventana de Entrada:** 168 horas (14 días solares).
- **Características:** Variables exógenas (Clima) + Codificación Cíclica (Hora y Mes).

#### Estrategia de Evaluación
Se utilizaron dos enfoques de validación:
1. **Simple Recursive (Ciego):** El modelo predice un bloque de 12h, inyecta su propia predicción como entrada para el siguiente bloque y así sucesivamente hasta completar el semestre (10% test). Es la prueba más difícil pues no hay corrección externa.
2. **Sliding Window (Día a Día):** El modelo predice 12h, pero recibe el dato real de generación para actualizar su memoria antes de predecir el siguiente día. Ideal para operación en tiempo real.

### 2. Modelos y Estrategias de Evaluación
Se aplicaron tres metodologías distintas para evaluar el desempeño en el periodo de test (Jul-Dic 2023):

| Modelo | Estrategia de Evaluación | Descripción |
| :--- | :--- | :--- |
| **SARIMAX** | **Recursive Forecast** | Predice paso a paso usando el clima real del futuro y su propia estructura de estacionalidad estadística. |
| **LSTM / MLP** | **Simple Recursive (Ciego)** | El modelo predice un paso y usa su propia salida como entrada para el siguiente. Es una prueba de resistencia a la deriva sin corrección real. |
| **XGBoost (Mem)** | **Autoregressive Recursive** | Utiliza una ventana de memoria (12m para mensual, 7d para diario) que se actualiza recursivamente con sus propias predicciones. |
| **XGBoost (Dir)** | **Direct Regression** | **Sin Memoria.** Predice la energía de un mes/día basándose exclusivamente en el clima de ese momento. No mira el historial de kWh. |

### 3. Resultados y Hallazgos Principales

#### **A. Comparativa Mensual (Cold-Start Jul-Dic 2023)**
En esta prueba, los modelos predijeron 6 meses basándose solo en el clima y su propio historial (si aplica).

| Modelo | MAPE (%) | RMSE | MAE | Estrategia |
| :--- | :---: | :---: | :---: | :--- |
| 🥇 **SARIMAX** | **8.56%** | 1,203,975 | 1,107,704 | Recursive |
| 🥈 **XGBoost (Mem)** | **12.92%** | 1,824,071 | 1,726,565 | Auto-Recursive |
| 🥉 **XGBoost (Dir)** | **14.85%** | 2,291,429 | 2,008,321 | **Direct Reg** |
| 4️⃣ **MLP** | 27.72% | 3,801,883 | 3,660,269 | Recursive |
| 5️⃣ **LSTM** | 38.81% | 5,328,785 | 5,128,718 | Recursive |

#### **B. Comparativa Diaria (Acumulado Mensual Jul-Dic 2023)**
Los modelos operaron a nivel horario y se acumuló el resultado para comparar con el cierre mensual.

| Modelo | MAPE (%) | RMSE | MAE | Hallazgo |
| :--- | :---: | :---: | :---: | :--- |
| 🥇 **SARIMAX** | **6.41%** | 1,223,760 | 872,761 | Líder en precisión horaria. |
| 🥈 **MLP** | **7.40%** | 1,028,498 | 957,490 | Mejor red neuronal en esta escala. |
| 🥉 **LSTM** | **8.99%** | 1,831,841 | 1,236,790 | Gran captura de estacionalidad. |
| 4️⃣ **XGBoost (Dir)** | 13.54% | 1,987,867 | 1,796,994 | Consistente con solo clima. |
| 5️⃣ **XGBoost (Mem)** | 23.37% | 3,178,524 | 3,066,689 | Acumulación de error recursivo. |

## Conclusión General
El proyecto demuestra que para el pronóstico de energía solar en esta planta:
1. Las variables climáticas (Irradiancia y Temperatura) son los predictores más críticos (explican >85% de la varianza).
2. Los modelos estadísticos (SARIMAX) son superiores para planeación estratégica de largo plazo (meses).
3. Las redes neuronales (MLP/LSTM) destacan en la operación táctica diaria, logrando errores menores al 10%.

---
*Desarrollado como parte del pipeline de optimización de pronósticos energéticos.*
