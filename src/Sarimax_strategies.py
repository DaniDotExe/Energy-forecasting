"""
SARIMAX: Comparación de Estrategias de Validación Temporal
==========================================================
Este script evalúa el modelo ganador SARIMAX(2,0,1)(1,1,1)_12 usando 3 enfoques
sobre el conjunto de validación (20% de los datos):

1. División Simple (Simple Split): Entrena en 70%, predice TODO el 20% de una vez (horizonte largo).
2. Ventana Expansiva (Expanding Window): Entrena en 70%. Día a día (12 horas),
   predice el día siguiente, y luego agrega los datos reales al modelo (actualiza el filtro).
3. Ventana Deslizante (Sliding Window): Entrena en 70%. Día a día, predice el día
   siguiente, pero la ventana de historia visible se mantiene del tamaño fijo del 70%.

Nota: Para mantener un tiempo de ejecución viable, las estrategias Expansiva y Deslizante
actualizan el estado del modelo (filtro de Kalman) día a día usando los parámetros ajustados
inicialmente, en lugar de re-estimar los coeficientes por máxima verosimilitud en cada paso.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'data-hourly.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'EDA', 'SARIMAX_training')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEASONAL_PERIOD = 12
EXOG_COLS = ['Irradiancia_It', 'Temperatura_Tt', 'Humedad_Ht', 'Viento_Wt']

# Modelo Ganador
ORDER = (2, 0, 1)
SEASONAL_ORDER = (1, 1, 1, SEASONAL_PERIOD)

# ──────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred, name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return {'Estrategia': name, 'RMSE': rmse, 'MAE': mae_val, 'MAPE (%)': mape_val}

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df = df.sort_values('Fecha_Hora').reset_index(drop=True)
    return df

def split_data(df):
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test

# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  COMPARACIÓN DE ESTRATEGIAS DE VALIDACIÓN TEMPORAL")
    print("=" * 70)
    
    df = load_data()
    train, val, test = split_data(df)
    
    train_ts = train['kWh']
    exog_train = train[EXOG_COLS]
    
    val_ts = val['kWh']
    exog_val = val[EXOG_COLS]
    
    n_train = len(train_ts)
    n_val = len(val_ts)
    
    print(f"  Entrenando modelo base SARIMAX{ORDER}{SEASONAL_ORDER} en datos de Train...")
    t0 = time.time()
    base_model = SARIMAX(
        train_ts,
        exog=exog_train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    base_result = base_model.fit(disp=False, maxiter=500)
    print(f"  Entrenamiento completado en {time.time() - t0:.1f}s\n")

    results_metrics = []
    predictions_dict = {}

    # ──────────────────────────────────────────────────────────────────
    # 1. DIVISIÓN SIMPLE (Simple Split)
    # ──────────────────────────────────────────────────────────────────
    print("  Ejecutando Estrategia 1: División Simple (1 solo forecast largo)...")
    t0 = time.time()
    forecast_simple = base_result.get_forecast(steps=n_val, exog=exog_val)
    y_pred_simple = forecast_simple.predicted_mean.values
    
    m_simple = compute_metrics(val_ts.values, y_pred_simple, "1. División Simple")
    results_metrics.append(m_simple)
    predictions_dict['Simple'] = y_pred_simple
    print(f"    RMSE: {m_simple['RMSE']:,.2f} | Tiempo: {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────
    # 2. VENTANA EXPANSIVA (Expanding Window)
    # ──────────────────────────────────────────────────────────────────
    print("\n  Ejecutando Estrategia 2: Ventana Expansiva (Paso de 1 día / 12h)...")
    t0 = time.time()
    
    y_pred_expanding = []
    current_res = base_result
    
    # Avanzamos día a día (bloques de 12 horas)
    step_size = SEASONAL_PERIOD
    
    for i in range(0, n_val, step_size):
        end_idx = min(i + step_size, n_val)
        current_step_size = end_idx - i
        
        # Exógenas para el horizonte a predecir
        exog_future = exog_val.iloc[i:end_idx]
        
        # Predicción
        fc = current_res.forecast(steps=current_step_size, exog=exog_future)
        y_pred_expanding.extend(fc.values)
        
        # Actualizamos el modelo añadiendo los datos reales (expandimos la historia)
        endog_new = val_ts.iloc[i:end_idx]
        exog_new = exog_val.iloc[i:end_idx]
        current_res = current_res.extend(endog_new, exog=exog_new)

    y_pred_expanding = np.array(y_pred_expanding)
    m_exp = compute_metrics(val_ts.values, y_pred_expanding, "2. Ventana Expansiva")
    results_metrics.append(m_exp)
    predictions_dict['Expansiva'] = y_pred_expanding
    print(f"    RMSE: {m_exp['RMSE']:,.2f} | Tiempo: {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────
    # 3. VENTANA DESLIZANTE (Sliding Window)
    # ──────────────────────────────────────────────────────────────────
    print("\n  Ejecutando Estrategia 3: Ventana Deslizante (Paso 1 día, ventana fija)...")
    t0 = time.time()
    
    y_pred_sliding = []
    
    # Preparamos el dataset completo para extraer ventanas fijas
    full_ts = pd.concat([train_ts, val_ts])
    full_exog = pd.concat([exog_train, exog_val])
    
    for i in range(0, n_val, step_size):
        end_idx = min(i + step_size, n_val)
        current_step_size = end_idx - i
        
        # Índice global en la serie completa donde termina el Train actual
        global_train_end = n_train + i
        
        # Ventana deslizante (mantiene tamaño = n_train)
        window_start = global_train_end - n_train
        window_endog = full_ts.iloc[window_start : global_train_end]
        window_exog = full_exog.iloc[window_start : global_train_end]
        
        # Exógenas para la predicción
        exog_future = full_exog.iloc[global_train_end : global_train_end + current_step_size]
        
        # Aplicamos parámetros a la nueva ventana sin reentrenar
        sliding_res = base_result.apply(window_endog, exog=window_exog)
        
        # Predicción
        fc = sliding_res.forecast(steps=current_step_size, exog=exog_future)
        y_pred_sliding.extend(fc.values)

    y_pred_sliding = np.array(y_pred_sliding)
    m_slid = compute_metrics(val_ts.values, y_pred_sliding, "3. Ventana Deslizante")
    results_metrics.append(m_slid)
    predictions_dict['Deslizante'] = y_pred_sliding
    print(f"    RMSE: {m_slid['RMSE']:,.2f} | Tiempo: {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────
    # TABLA COMPARATIVA
    # ──────────────────────────────────────────────────────────────────
    df_metrics = pd.DataFrame(results_metrics)
    csv_path = os.path.join(OUTPUT_DIR, 'comparacion_estrategias.csv')
    df_metrics.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("  RESULTADOS DE ESTRATEGIAS")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print(f"\nGuardado en: {csv_path}")

    # ──────────────────────────────────────────────────────────────────
    # GRÁFICAS COMPARATIVAS
    # ──────────────────────────────────────────────────────────────────
    # 1. Gráfica de Métricas (Barras)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['RMSE', 'MAE', 'MAPE (%)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for ax, metric, color in zip(axes, metrics, colors):
        ax.bar(df_metrics['Estrategia'], df_metrics[metric], color=color, edgecolor='black', alpha=0.8)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor')
        ax.tick_params(axis='x', rotation=15)
        for i, val_metric in enumerate(df_metrics[metric]):
            ax.text(i, val_metric, f'{val_metric:,.1f}', ha='center', va='bottom', fontweight='bold')
            
    plt.tight_layout()
    metrics_plot_path = os.path.join(OUTPUT_DIR, 'estrategias_metricas.png')
    plt.savefig(metrics_plot_path, dpi=200)
    plt.close()

    # 2. Gráfica de Serie de Tiempo (Detalle último mes de validación para ver diferencias)
    last_month_idx = slice(-360, None)  # Últimos 30 días aprox
    val_dates = val['Fecha_Hora']
    
    plt.figure(figsize=(15, 6))
    plt.plot(val_dates.iloc[last_month_idx], val_ts.values[last_month_idx], label='Real', color='black', alpha=0.7, linewidth=1.5)
    
    plt.plot(val_dates.iloc[last_month_idx], predictions_dict['Simple'][last_month_idx], label='Simple Split', alpha=0.8, linewidth=1)
    plt.plot(val_dates.iloc[last_month_idx], predictions_dict['Expansiva'][last_month_idx], label='Ventana Expansiva', alpha=0.8, linewidth=1)
    plt.plot(val_dates.iloc[last_month_idx], predictions_dict['Deslizante'][last_month_idx], label='Ventana Deslizante', alpha=0.8, linewidth=1)
    
    plt.title('Comparación de Predicciones - Último Mes de Validación', fontsize=14, fontweight='bold')
    plt.ylabel('kWh')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    series_plot_path = os.path.join(OUTPUT_DIR, 'estrategias_series.png')
    plt.savefig(series_plot_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    main()
