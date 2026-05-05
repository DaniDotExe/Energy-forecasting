# Proyecto de Pronóstico de Energía Solar - Planta "El Paso"

Este proyecto contiene una serie de scripts en Python diseñados para la extracción, procesamiento y consolidación de datos climáticos y de generación eléctrica para la planta solar "El Paso" (Cesar, Colombia). El objetivo final es generar datasets limpios para realizar predicciones de la generación de energía solar con diferentes modelos de aprendizaje automático.

Ubicación de la planta: 9.785041627157241, -73.7201783147516

Rango de fecha de los datos: 2019-01-01 a 2023-12-31

## Fuentes de Datos

El proyecto utiliza actualmente tres fuentes principales de información (en desarrollo):

### 1. Open-Meteo (Datos Climáticos de Reanálisis)
Se utiliza la API de [Open-Meteo](https://open-meteo.com/) (Modelo ERA5/ERA5-Land) para obtener el histórico climático en las coordenadas exactas de la planta. Los datos están configurados para el rango horario de **06:00 a 17:00** (12 horas por día).

| Variable | Columna CSV | Unidad | Descripción |
| :--- | :--- | :--- | :--- |
| **Fecha y Hora** | `Fecha_Hora` | YYYY-MM-DD HH:MM | Marca de tiempo en zona horaria local. |
| **Irradiancia** | `Irradiancia_It` | W/m² | Irradiancia de onda corta global. |
| **Temperatura** | `Temperatura_Tt` | °C | Temperatura del aire a 2 metros de altura. |
| **Humedad** | `Humedad_Ht` | % | Humedad relativa a 2 metros de altura. |
| **Viento** | `Viento_Wt` | km/h | Velocidad del viento a 10 metros de altura. |

**Archivo de salida:** `data/openmeteo-horario.csv`

### 2. NASA POWER (Datos Climáticos Satelitales)
Se utiliza la API de [NASA POWER](https://power.larc.nasa.gov/) (Prediction Of Worldwide Energy Resources) para obtener un segundo conjunto de reanálisis climático para las mismas coordenadas y el mismo rango horario diurno (**06:00 a 17:00**).

| Variable API | Columna CSV | Unidad | Descripción |
| :--- | :--- | :--- | :--- |
| `time_local` | `Fecha_Hora` | YYYY-MM-DD HH:MM | Marca de tiempo en zona horaria local. |
| `ALLSKY_SFC_SW_DWN` | `Irradiancia_It` | Wh/m² | Irradiancia de onda corta hacia abajo (superficie). |
| `T2M` | `Temperatura_Tt` | °C | Temperatura a 2 metros. |
| `RH2M` | `Humedad_Ht` | % | Humedad relativa a 2 metros. |
| `WS10M` | `Viento_Wt` | km/h | Velocidad del viento a 10 metros (convertido de m/s). |

**Archivo de salida:** `data/nasapower-horario.csv`

---
*Próximamente se documentarán las fuentes de Generación Real (XM).*
