# Proyecto de Pronóstico de Energía Solar - Planta "El Paso"

Este proyecto contiene una serie de scripts en Python diseñados para la extracción, procesamiento y consolidación de datos climáticos y de generación eléctrica para la planta solar "El Paso" (Cesar, Colombia). El objetivo final es generar datasets limpios (`daily_data.csv` y `monthly_data.csv`) listos para ser utilizados en modelos de Redes Neuronales Recurrentes (RNN).

## Origen y Naturaleza de los Datos

Para el entrenamiento del modelo, se han combinado dos fuentes de datos distintas:

*   **Datos de Generación (kWh):** Son **datos reales y medidos**. Provienen de los registros oficiales de **XM** (operador del mercado eléctrico colombiano). Representan la energía real captada por los sensores y medidores físicos de la planta solar "El Paso".
*   **Datos Climáticos (Temperatura, Viento, Irradiancia):** Son **datos de reanálisis meteorológico** obtenidos a través de la API de **Open-Meteo** (Modelo ERA5). Proporcionan una estimación científica de alta precisión de las condiciones atmosféricas en las coordenadas exactas de la planta, integrando información satelital y estaciones meteorológicas globales.

## Estructura de Scripts

### 1. `API_OpenMeteo_Solar.py`
Script original que descarga el histórico climático desde la API de Open-Meteo. 
- **Rango:** 2019-01-01 a 2023-12-31.
- **Filtro:** Horas diurnas (6:00 a.m. a 5:00 p.m.).
- **Agregación:** Promedios mensuales.


### 2. `clima_diario_el_paso_diurno.py`
Versión adaptada del script de API para obtener datos con granularidad diaria.
- **Función:** Descarga los mismos parámetros climáticos pero calcula el promedio diario basándose únicamente en el rango de luz solar (6:00 a.m. a 5:00 p.m.).
- **Salida:** `clima_diario_el_paso_diurno.csv`.

### 3. `extract_el_paso_generation.py`
Script encargado de procesar los archivos de generación de XM (formato Excel) de los años 2019 a 2023.
- **Filtros:** Extrae únicamente la planta `"EL PASO"` con tipo de combustible `"RAD SOLAR"`.
- **Acción:** Realiza un "stack" (concatenación vertical) de los 5 años de datos.
- **Salida:** `data_2019_2023_kwH_xm.csv`.

### 4. `create_final_dataset.py`
Script de consolidación final que une los datos de generación y clima.
- **Acción:** Realiza un `merge` interno basado en la fecha, extrae el **Año**, **Mes** y **Día**, y selecciona el rango de horas de generación de las 6 a las 17 (5 p.m.). También genera automáticamente el agregado mensual.
- **Salida:** `daily_data.csv` y `monthly_data.csv`.

---

## Resumen de los Datasets Finales

Los archivos resultantes son el producto final del pipeline de datos y contienen la información consolidada necesaria para el entrenamiento del modelo.

### 1. `daily_data.csv` (Granularidad Diaria)
- **Total de registros:** 1,811 días (cobertura completa de 2019 a 2023).
- **Columnas Incluidas:**
  1.  **`Fecha`**: Fecha del registro (YYYY-MM-DD).
  2.  **`Año`**: Año del registro (2019-2023).
  3.  **`Mes`**: Mes del año (1-12).
  4.  **`Dia`**: Día del mes (1-31).
  5.  **`Recurso`**: Nombre de la planta ("EL PASO").
  6.  **`6` al `17`**: Generación real en kWh para cada una de las 12 horas diurnas.
  7.  **`Total_Generacion`**: Suma de la generación (de las horas 6 a 17) dando el total de ese día.
  8.  **`Temperatura_Tt`**: Temperatura promedio diaria (°C).
  9.  **`Viento_Wt`**: Velocidad del viento promedio diaria (km/h).
  10. **`Irradiancia_It`**: Irradiancia de onda corta promedio diaria (W/m²).

### 2. `monthly_data.csv` (Granularidad Mensual)
- **Total de registros:** 60 meses (5 años * 12 meses).
- **Columnas Incluidas:**
  1.  **`Año`** y **`Mes`**: Identificadores del mes.
  2.  **`Recurso`**: Nombre de la planta ("EL PASO").
  3.  **`6` al `17`**: Promedio mensual de generación en kWh para cada una de esas horas.
  4.  **`Total_Generacion`**: Promedio global mensual de la generación total diaria.
  5.  **`Temperatura_Tt`, `Viento_Wt`, `Irradiancia_It`**: Promedio mensual de las variables climáticas.

---
*Este dataset está optimizado para capturar la variabilidad estacional y diaria de la generación solar en relación con las condiciones climáticas locales.*
