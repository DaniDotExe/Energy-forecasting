# Proyecto de Pronóstico de Energía Solar - Planta "El Paso"

Este proyecto contiene una serie de scripts en Python diseñados para la extracción, procesamiento y consolidación de datos climáticos y de generación eléctrica para la planta solar "El Paso" (Cesar, Colombia). El objetivo final es generar datasets limpios para realizar predicciones de la generación de energía solar con diferentes modelos de aprendizaje automático.

Ubicación de la planta: 9.785041627157241, -73.7201783147516

Rango de fecha de los datos: 2019-01-01 a 2023-12-31

## Origen y Naturaleza de los Datos

Para el entrenamiento del modelo, se han combinado dos fuentes de datos distintas:

*   **Datos de Generación (kWh):** Son **datos reales y medidos**. Provienen de los registros oficiales de **XM** (operador del mercado eléctrico colombiano). Representan la energía real captada por los sensores y medidores físicos de la planta solar "El Paso".
*   **Datos Climáticos (Temperatura, Viento, Irradiancia):** Son **datos de reanálisis 

