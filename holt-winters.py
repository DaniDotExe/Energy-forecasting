#!/usr/bin/env python3
"""
holt-winters.py

Modelo estadístico Holt-Winters (Suavizado Exponencial) para predecir la generación solar.
Procesa tanto datos diarios como mensuales. A diferencia de la red neuronal, este modelo 
es univariado y solo se basa en el histórico de "Total_Generacion", detectando
su tendencia y estacionalidad.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
BASE_DIR = r"d:\Software\Energy-forecasting"
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "resultados_hw")
TARGET_COL = 'Total_Generacion'

def run_hw_pipeline(data_path, output_subdir, seasonal_periods, title_prefix):
    """Ejecuta el pipeline Holt-Winters para un dataset específico."""
    print(f"\n>>> PROCESANDO: {title_prefix} <<<")
    
    # Crear carpeta de resultados
    out_dir = os.path.join(RESULTS_BASE_DIR, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Cargar datos
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # Ajuste de nombres de columnas en caso de caracteres extraños
    if 'Año' not in df.columns:
        df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
        year_col = [c for c in df.columns if 'o' in c.lower() and len(c) <= 5]
        if year_col: df.rename(columns={year_col[0]: 'Año'}, inplace=True)
    
    df['Fecha'] = pd.to_datetime(df['Fecha']) if 'Fecha' in df.columns else pd.to_datetime(df['Año'].astype(str)+'-'+df['Mes'].astype(str)+'-01')
    df = df.sort_values('Fecha').reset_index(drop=True)

    y = df[TARGET_COL].values
    fechas = df['Fecha'].values

    # 2. Split 90/10 (Train+Val para ajustar, Test para evaluar)
    # Utilizamos el 90% inicial para alinear equitativamente con la métrica del RNN (Train+Val)
    n = len(y)
    train_val_split = int(n * 0.90)

    y_train_val = y[:train_val_split]
    y_test = y[train_val_split:]
    
    fechas_train_val = fechas[:train_val_split]
    fechas_test = fechas[train_val_split:]

    # 3. Modelo Holt-Winters
    # trend='add': tendencia aditiva. seasonal='add': estacionalidad aditiva.
    # Evitamos estacionalidad multiplicativa porque puede fallar si hay ceros.
    model = ExponentialSmoothing(
        y_train_val, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=seasonal_periods,
        initialization_method="estimated"
    )
    fitted_model = model.fit()

    # 4. Predicciones
    # In-sample (ajuste sobre los datos de entrenamiento)
    y_train_val_pred = fitted_model.fittedvalues
    
    # Out-of-sample (Test - Pronóstico)
    y_test_pred = fitted_model.forecast(len(y_test))

    # 5. Métricas (MAPE)
    # Prevenimos divisiones por cero con un pequeño epsilon
    epsilon = 1e-10
    mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + epsilon))) * 100

    # 6. Graficar
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Datos reales
    ax.plot(fechas_train_val, y_train_val, 'b-', label='Real Train/Val', alpha=0.3)
    ax.plot(fechas_test, y_test, 'k-', label='Real Test', alpha=0.3)
    
    # Predicciones
    ax.plot(fechas_train_val, y_train_val_pred, 'r--', label='Ajuste Train/Val', alpha=0.8)
    ax.plot(fechas_test, y_test_pred, 'c--', label='Predicción Test', alpha=0.8)
    
    ax.set_title(f'Predicción Holt-Winters ({title_prefix}) - MAPE Test: {mape:.2f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'prediccion_completa_hw.png'), dpi=300)
    plt.close()

    print(f"  Resultados en {output_subdir}: MAPE = {mape:.2f}%")
    return mape

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    experimentos = [
        {
            'path': os.path.join(BASE_DIR, "monthly_data.csv"),
            'subdir': 'monthly',
            'seasonal_periods': 12, # Estacionalidad anual mensual (12 meses)
            'name': 'Datos Mensuales'
        },
        {
            'path': os.path.join(BASE_DIR, "daily_data.csv"),
            'subdir': 'daily',
            'seasonal_periods': 365, # Estacionalidad anual diaria (365 días)
            'name': 'Datos Diarios'
        }
    ]

    for exp in experimentos:
        try:
            run_hw_pipeline(exp['path'], exp['subdir'], exp['seasonal_periods'], exp['name'])
        except Exception as e:
            print(f"Error procesando {exp['name']}: {e}")

    print("\n=== Proceso completo finalizado. Revisa la carpeta 'resultados_hw' ===")
