import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE DIRECTORIOS Y LIBRERÍAS
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_HOURLY_PATH = os.path.join(BASE_DIR, '..', 'data', 'data-hourly.csv')
DATA_MONTHLY_PATH = os.path.join(BASE_DIR, '..', 'data', 'monthly-kwh.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'EDA', 'FINAL_COMPARISON')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Importamos las funciones y clases del modelo LSTM
sys.path.insert(0, BASE_DIR)
from LSTM_model import (
    get_device, fit_scalers, scale_data, create_sequences, 
    SolarDataset, SolarLSTM, train_model, predict_sequences, 
    inverse_scale_target, compute_metrics, add_cyclic_features,
    INPUT_WINDOW, OUTPUT_WINDOW, BATCH_SIZE, ALL_FEATURES, N_FEATURES,
    EXOG_COLS
)
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────
def load_and_split_data():
    print("-" * 70)
    print("  CARGA Y DIVISIÓN DE DATOS (90% Train / 10% Test)")
    print("-" * 70)
    
    # Carga datos por hora
    df_hourly = pd.read_csv(DATA_HOURLY_PATH)
    df_hourly['Fecha_Hora'] = pd.to_datetime(df_hourly['Fecha_Hora'])
    df_hourly = df_hourly.sort_values('Fecha_Hora').reset_index(drop=True)
    df_hourly = add_cyclic_features(df_hourly)
    
    # Para cuadrar exactamente con los meses y el 10% (aprox 2174 registros), 
    # cortaremos en 2023-07-01. Así el test set son exactamente 6 meses (Jul-Dic 2023)
    # 6 meses * ~30 días * 12 horas = ~2160 registros (aprox 10%)
    split_date = pd.to_datetime('2023-07-01 00:00:00')
    
    train_df = df_hourly[df_hourly['Fecha_Hora'] < split_date].copy()
    test_df = df_hourly[df_hourly['Fecha_Hora'] >= split_date].copy()
    
    print(f"  Total registros: {len(df_hourly)}")
    print(f"  Train (90%): {len(train_df)} ({train_df['Fecha_Hora'].min()} -> {train_df['Fecha_Hora'].max()})")
    print(f"  Test  (10%):  {len(test_df)} ({test_df['Fecha_Hora'].min()} -> {test_df['Fecha_Hora'].max()})")
    
    # Carga datos mensuales
    df_monthly = pd.read_csv(DATA_MONTHLY_PATH)
    
    return train_df, test_df, df_monthly

def run_sarimax(train_df, test_df):
    print("\n" + "-" * 70)
    print("  ENTRENAMIENTO Y PREDICCIÓN SARIMAX(2,0,1)(1,1,1)_12")
    print("-" * 70)
    t0 = time.time()
    
    endog_train = train_df['kWh']
    exog_train = train_df[EXOG_COLS]
    exog_test = test_df[EXOG_COLS]
    
    print("  Ajustando modelo SARIMAX (esto puede tardar unos minutos)...")
    model = SARIMAX(
        endog_train,
        exog=exog_train,
        order=(2, 0, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)
    print(f"  Entrenamiento SARIMAX completado en {time.time() - t0:.1f}s")
    
    print("  Generando predicciones sobre el 10% (Test set)...")
    preds = fitted_model.get_forecast(steps=len(test_df), exog=exog_test).predicted_mean
    
    # Asegurarnos de que no haya negativos
    preds = np.maximum(preds, 0)
    return preds.values

def run_lstm(train_df, test_df, device):
    print("\n" + "-" * 70)
    print("  ENTRENAMIENTO Y PREDICCIÓN LSTM (Ventana 14 días)")
    print("-" * 70)
    t0 = time.time()
    
    # Para el LSTM, necesitamos un split interno de validación para el EarlyStopping
    # Usaremos el último 10% del train_df como validación interna
    n_internal_train = int(len(train_df) * 0.9)
    internal_train_df = train_df.iloc[:n_internal_train].copy()
    internal_val_df = train_df.iloc[n_internal_train:].copy()
    
    scaler = fit_scalers(internal_train_df)
    scaled_train = scale_data(internal_train_df, scaler)
    scaled_val = scale_data(internal_val_df, scaler)
    scaled_test = scale_data(test_df, scaler)
    
    # Secuencias
    X_train, y_train = create_sequences(scaled_train)
    X_val, y_val = create_sequences(scaled_val)
    
    train_loader = DataLoader(SolarDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SolarDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # Modelo
    model = SolarLSTM()
    print(f"  Arquitectura LSTM lista con {sum(p.numel() for p in model.parameters()):,} parámetros.")
    
    # Entrenamiento
    model, _ = train_model(model, train_loader, val_loader, device)
    
    # Predicción Recursiva (Estrategia 1) sobre el test_df
    print("\n  Generando predicciones recursivas sobre el 10% (Test set)...")
    
    # El contexto inicial es el final de train_df (que incluye internal_val)
    # Por lo que unimos internal_train + internal_val para tener todo el historial 
    full_scaled_train = np.concatenate([scaled_train, scaled_val], axis=0)
    current_context = full_scaled_train[-INPUT_WINDOW:].copy()
    
    n_test = len(scaled_test)
    preds_scaled = []
    
    # Para iterar de forma recursiva
    for i in range(0, n_test - OUTPUT_WINDOW + 1, OUTPUT_WINDOW):
        # 1. Preparar secuencia
        seq_x = current_context[-INPUT_WINDOW:].reshape(1, INPUT_WINDOW, N_FEATURES)
        
        # 2. Predecir
        pred_kwh_scaled = predict_sequences(model, seq_x, device)[0]
        preds_scaled.append(pred_kwh_scaled)
        
        # 3. Actualizar contexto
        next_block = scaled_test[i : i + OUTPUT_WINDOW].copy()
        next_block[:, -1] = pred_kwh_scaled
        current_context = np.concatenate([current_context, next_block], axis=0)
        
    preds_flat = np.concatenate(preds_scaled)
    
    # Des-escalar
    preds_kwh = inverse_scale_target(preds_flat, scaler)
    
    # Asegurar no negativos
    preds_kwh = np.maximum(preds_kwh, 0)
    
    print(f"  Proceso LSTM completado en {time.time() - t0:.1f}s")
    return preds_kwh

def plot_monthly_comparison(df_monthly_test, output_dir):
    print("\n" + "-" * 70)
    print("  GENERANDO GRÁFICA MENSUAL (REAL VS SARIMAX VS LSTM)")
    print("-" * 70)
    
    meses = df_monthly_test['Mes'].values
    real = df_monthly_test['Real'].values
    sarimax = df_monthly_test['SARIMAX'].values
    lstm = df_monthly_test['LSTM'].values
    
    x = np.arange(len(meses))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, real, width, label='Real (monthly-kwh.csv)', color='#2ca02c')
    bars2 = ax.bar(x, sarimax, width, label='Predicción SARIMAX', color='#1f77b4')
    bars3 = ax.bar(x + width, lstm, width, label='Predicción LSTM', color='#ff7f0e')
    
    ax.set_ylabel('Generación Total (kWh)', fontsize=11, fontweight='bold')
    ax.set_title('Comparación Mensual de Generación Solar (Test Set: Jul-Dic 2023)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(meses, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Función para formatear las etiquetas en millones (M)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height/1e6:.1f}M',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'comparacion_mensual_final.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Gráfica guardada en: {filepath}")

def main():
    print("=" * 70)
    print("  BATALLA FINAL: SARIMAX vs LSTM")
    print("=" * 70)
    
    device = get_device()
    train_df, test_df, df_monthly_raw = load_and_split_data()
    
    # 1. Correr modelos
    sarimax_preds = run_sarimax(train_df, test_df)
    lstm_preds = run_lstm(train_df, test_df, device)
    
    # Asignar predicciones al DataFrame horario de test
    test_df['SARIMAX_Pred'] = sarimax_preds
    test_df['LSTM_Pred'] = lstm_preds
    
    # 2. Agrupar predicciones por Mes
    # Extraemos el string 'YYYY-MM'
    test_df['Mes'] = test_df['Fecha_Hora'].dt.strftime('%Y-%m')
    
    monthly_preds = test_df.groupby('Mes')[['SARIMAX_Pred', 'LSTM_Pred']].sum().reset_index()
    
    # 3. Filtrar el df_monthly_raw para que solo tenga los meses del test set
    test_months = monthly_preds['Mes'].unique()
    df_monthly_real = df_monthly_raw[df_monthly_raw['Mes'].isin(test_months)].copy()
    
    # Hacemos merge para alinear los datos
    df_comparison = pd.merge(df_monthly_real, monthly_preds, on='Mes')
    df_comparison = df_comparison.rename(columns={
        'Generacion_Total_kWh': 'Real',
        'SARIMAX_Pred': 'SARIMAX',
        'LSTM_Pred': 'LSTM'
    })
    
    print("\n" + "-" * 70)
    print("  RESULTADOS MENSUALES DE LA BATALLA FINAL")
    print("-" * 70)
    print(df_comparison.to_string(index=False))
    
    # Guardar a CSV
    csv_path = os.path.join(OUTPUT_DIR, 'resultados_mensuales.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n  Resultados guardados en: {csv_path}")
    
    # Calcular error absoluto total en el semestre
    total_real = df_comparison['Real'].sum()
    total_sarimax = df_comparison['SARIMAX'].sum()
    total_lstm = df_comparison['LSTM'].sum()
    
    err_sarimax = abs(total_sarimax - total_real) / total_real * 100
    err_lstm = abs(total_lstm - total_real) / total_real * 100
    
    print(f"\n  Error Total del Semestre (Jul-Dic 2023):")
    print(f"    SARIMAX : {err_sarimax:.2f}% de desvío")
    print(f"    LSTM    : {err_lstm:.2f}% de desvío")
    
    # 4. Generar gráfica
    plot_monthly_comparison(df_comparison, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("  BATALLA COMPLETADA CON ÉXITO")
    print("=" * 70)

if __name__ == '__main__':
    main()
