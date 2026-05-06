# Pronóstico de Energía Solar - Planta "El Paso" (Cesar, Colombia)

Este proyecto implementa un pipeline avanzado para el análisis y pronóstico de generación de energía solar. Todo el proceso, desde la ingesta de datos hasta la comparación de modelos de aprendizaje profundo y estadísticos, está consolidado en el notebook `Energy-forecasting.ipynb`.

## 📍 Ubicación y Alcance
- **Planta**: Solar "El Paso" (Cesar, Colombia).
- **Coordenadas**: 9.785041627157241, -73.7201783147516.
- **Periodo de datos**: 2019-01-01 a 2023-12-31.

## 📊 Fuentes de Datos y Variables
El modelo utiliza una combinación de datos reales y satelitales:
1. **Generación Real (XM)**: Datos horarios de telemedida oficial (kWh) del operador del sistema en Colombia.
2. **Clima (NASA POWER)**: Seleccionado por su integridad (0% valores nulos) frente a otras fuentes como Open-Meteo.
    - **Variables**: Irradiancia (`Irradiancia_It`), Temperatura (`Temperatura_Tt`) y Viento (`Viento_Wt`).
    - **Optimización**: La variable de Humedad (`Humedad_Ht`) fue descartada tras un análisis de correlación por presentar redundancia (multicolinealidad) con la temperatura.

## 🛠️ Metodología de Procesamiento
- **Filtro de Horas Diurnas**: Solo se conservan registros entre **06:00 AM y 05:00 PM** para evitar sesgos por generación nula nocturna.
- **Preprocesamiento**: Normalización Z-score y división de datos 80/20 (entrenamiento/prueba). El último año de datos se utiliza para la validación final.

## 📈 Análisis y Modelado

### 1. Análisis Estadístico (SARIMAX)
Se justifica la elección del modelo mediante pruebas de estacionariedad (ADF) y análisis de autocorrelación (ACF/PACF).
- **Modelo Final**: SARIMAX (2,0,1)(1,1,1)₁₂ o (1,0,1)(0,1,1)₁₂ según la escala de agregación.
- **Justificación**: Captura tanto la dependencia autoregresiva inmediata como la estacionalidad cíclica de la generación solar.

### 2. Deep Learning (LSTM)
Implementación de una red neuronal recurrente (LSTM) diseñada para capturar patrones secuenciales complejos.
- **Optimización**: Búsqueda en rejilla (Grid Search) sobre `hidden_size`, `num_layers`, `learning_rate` y `input_window`.
- **Entrenamiento**: 1000 épocas con paciencia de 100 (Early Stopping) para evitar el sobreajuste.

## 🏁 Evaluación de Desempeño
El desempeño se mide comparando el **SARIMAX vs. LSTM** mediante métricas estándar:
- **RMSE** (Error Cuadrático Medio)
- **MAE** (Error Absoluto Medio)
- **MAPE** (Error Porcentual Absoluto Medio)

El pipeline genera automáticamente:
- Gráfica de **Loss** (Entrenamiento vs. Validación).
- Comparativa visual de **Predicción vs. Real** mes a mes.
- Gráfica de **Error Porcentual por Mes**.
- Análisis de **Residuos** del modelo estadístico.

## 📁 Artifactos de Salida
La ejecución del notebook produce los siguientes archivos de resultados:
- `model_comparison_metrics.csv`: Tabla con el resumen de errores de cada modelo.
- `monthly_comparison_table.csv`: Tabla detallada de valores Reales vs. Predichos.
- Imágenes de visualización (`01_lstm_loss.png`, `02_prediccion_vs_real.png`, etc.).

---
*Este proyecto proporciona una herramienta robusta y portátil para la predicción de generación renovable en entornos cloud (Google Colab).*
