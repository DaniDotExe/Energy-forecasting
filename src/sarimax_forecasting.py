#!/usr/bin/env python3
"""
sarimax_forecasting.py

Modelo estadístico SARIMAX para predecir la generación solar.
Procesa tanto datos diarios como mensuales. A diferencia de Holt-Winters,
SARIMAX permite usar las variables exógenas (Clima) al igual que la Red Neuronal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
BASE_DIR = r"d:\Software\Energy-forecasting"
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "output", "resultados_sarimax")

FEATURE_COLS = ['Temperatura_Tt', 'Viento_Wt', 'Irradiancia_It']
TARGET_COL = 'Total_Generacion'

def run_sarimax_pipeline(data_path, output_subdir, order, seasonal_order, title_prefix):
    """Ejecuta el pipeline SARIMAX para un dataset específico."""
    print(f"\n>>> PROCESANDO: {title_prefix} <<<")
    
    # Crear carpeta de resultados
    out_dir = os.path.join(RESULTS_BASE_DIR, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Cargar datos
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    if 'Año' not in df.columns:
        df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
        year_col = [c for c in df.columns if 'o' in c.lower() and len(c) <= 5]
        if year_col: df.rename(columns={year_col[0]: 'Año'}, inplace=True)
    
    df['Fecha'] = pd.to_datetime(df['Fecha']) if 'Fecha' in df.columns else pd.to_datetime(df['Año'].astype(str)+'-'+df['Mes'].astype(str)+'-01')
    df = df.sort_values('Fecha').reset_index(drop=True)

    # 2. Preparar Variables (Target y Exógenas)
    y = df[TARGET_COL].values
    X_exog = df[FEATURE_COLS].values
    fechas = df['Fecha'].values

    # Escalar variables exógenas (ayuda a la convergencia matemática del SARIMAX)
    scaler_X = StandardScaler()
    X_exog_scaled = scaler_X.fit_transform(X_exog)

    # 3. Split 90/10 (Train+Val para ajustar, Test para evaluar)
    n = len(y)
    train_val_split = int(n * 0.90)

    y_train_val = y[:train_val_split]
    y_test = y[train_val_split:]
    
    X_train_val = X_exog_scaled[:train_val_split]
    X_test = X_exog_scaled[train_val_split:]
    
    fechas_train_val = fechas[:train_val_split]
    fechas_test = fechas[train_val_split:]

    # 4. Modelo SARIMAX
    # Usamos enforce_stationarity y enforce_invertibility en False para evitar errores 
    # de convergencia, especialmente comunes en datos diarios
    print(f"  Ajustando modelo SARIMAX orden={order}, estacionalidad={seasonal_order}...")
    model = SARIMAX(
        endog=y_train_val, 
        exog=X_train_val, 
        order=order, 
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False, maxiter=200)

    # 5. Predicciones
    # In-sample (ajuste sobre los datos de entrenamiento)
    y_train_val_pred = fitted_model.fittedvalues
    
    # Out-of-sample (Test - Pronóstico usando las exógenas futuras)
    y_test_pred = fitted_model.forecast(steps=len(y_test), exog=X_test)

    # 6. Métricas (MAPE)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + epsilon))) * 100

    # 7. Graficar
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Datos reales
    ax.plot(fechas_train_val, y_train_val, 'b-', label='Real Train/Val', alpha=0.3)
    ax.plot(fechas_test, y_test, 'k-', label='Real Test', alpha=0.3)
    
    # Predicciones
    ax.plot(fechas_train_val, y_train_val_pred, 'r--', label='Ajuste Train/Val', alpha=0.8)
    ax.plot(fechas_test, y_test_pred, 'c--', label='Predicción Test', alpha=0.8)
    
    ax.set_title(f'Predicción SARIMAX con variables Climáticas ({title_prefix}) - MAPE Test: {mape:.2f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'prediccion_completa_sarimax.png'), dpi=300)
    plt.close()

    # 8. Graficar (Solo Test)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fechas_test, y_test, 'b-o', label='Real', markersize=4, alpha=0.7)
    ax.plot(fechas_test, y_test_pred, 'r--s', label='Predicción', markersize=4, alpha=0.8)
    ax.set_title(f'Predicción Test SARIMAX ({title_prefix}) - MAPE: {mape:.2f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'prediccion_test_sarimax.png'), dpi=300)
    plt.close()

    print(f"  Resultados en {output_subdir}: MAPE = {mape:.2f}%")
    return mape

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # La configuración del orden (p, d, q) x (P, D, Q, m) es empírica
    # Para obtener parámetros perfectos usualmente se usa un auto_arima grid search
    experimentos = [
        {
            'path': os.path.join(BASE_DIR, "data", "monthly_data.csv"),
            'subdir': 'monthly',
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 0, 12), # Estacionalidad anual mensual (12 meses)
            'name': 'Datos Mensuales'
        },
        {
            'path': os.path.join(BASE_DIR, "data", "daily_data.csv"),
            'subdir': 'daily',
            'order': (1, 1, 1),
            # Para diarios usar m=365 requiere extrema computación. Usaremos m=7 (semanal) como 
            # aproximación, dado que la irradiancia tiene patrones influenciados pero a corto plazo.
            'seasonal_order': (1, 0, 1, 7), 
            'name': 'Datos Diarios'
        }
    ]

    for exp in experimentos:
        try:
            run_sarimax_pipeline(exp['path'], exp['subdir'], exp['order'], exp['seasonal_order'], exp['name'])
        except Exception as e:
            print(f"Error procesando {exp['name']}: {e}")

    print("\n=== Proceso completo finalizado. Revisa la carpeta 'resultados_sarimax' ===")
