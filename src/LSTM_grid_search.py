import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN Y RUTAS
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'data-hourly.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'EDA', 'LSTM_grid_search')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Importamos las funciones base de LSTM_model.py
sys.path.insert(0, BASE_DIR)
from LSTM_model import (
    get_device, load_data, split_data, fit_scalers, scale_data,
    create_sequences, SolarDataset, SolarLSTM, train_model,
    predict_sequences, inverse_scale_target, compute_metrics,
    OUTPUT_WINDOW, BATCH_SIZE, ALL_FEATURES, N_FEATURES
)

# ──────────────────────────────────────────────────────────────────────
# ESTRATEGIA DE EVALUACIÓN: VENTANA DESLIZANTE
# ──────────────────────────────────────────────────────────────────────
def evaluate_sliding_window(model, scaled_val, scaler, device, input_window):
    """
    Estrategia Deslizante: Usa historial real para cada predicción de 12h.
    """
    n_val = len(scaled_val)
    preds_scaled = []
    actuals_scaled = []
    
    # Para la ventana deslizante sobre el validation set, necesitamos
    # el final del entrenamiento como 'piso' inicial.
    # Pero para simplificar en el grid, evaluaremos sobre bloques del val_set.
    # Empezamos desde input_window hasta el final.
    for i in range(0, n_val - OUTPUT_WINDOW + 1, OUTPUT_WINDOW):
        if i < input_window: continue # No hay suficiente historial todavía
        
        # 1. Tomamos el historial REAL inmediatamente anterior
        seq_x = scaled_val[i - input_window : i].reshape(1, input_window, N_FEATURES)
        
        # 2. Predecimos las próximas 12h
        pred_kwh_scaled = predict_sequences(model, seq_x, device)[0]
        
        preds_scaled.append(pred_kwh_scaled)
        actuals_scaled.append(scaled_val[i : i + OUTPUT_WINDOW, -1])

    if not preds_scaled: return None, None

    preds_flat = np.concatenate(preds_scaled)
    actuals_flat = np.concatenate(actuals_scaled)
    
    preds_kwh = inverse_scale_target(preds_flat, scaler)
    actuals_kwh = inverse_scale_target(actuals_flat, scaler)
    
    return actuals_kwh, preds_kwh

# ──────────────────────────────────────────────────────────────────────
# GRID SEARCH MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  LSTM GRID SEARCH - COMPARACIÓN DE VENTANAS DE ENTRADA")
    print("=" * 70)
    
    device = get_device()
    df = load_data(DATA_PATH)
    train_df, val_df, test_df = split_data(df)
    
    # Configuración de la Grid
    # 7 días = 84h, 14 días = 168h, 30 días = 360h
    grid_windows = {
        '7 Días (84h)': 84,
        '14 Días (168h)': 168,
        '30 Días (360h)': 360
    }
    
    results = []
    all_predictions = {}

    for name, window_size in grid_windows.items():
        print(f"\n>>> ENTRENANDO MODELO: {name} (Ventana={window_size})")
        print("-" * 50)
        
        # 1. Preparar datos específicos para esta ventana
        scaler = fit_scalers(train_df)
        scaled_train = scale_data(train_df, scaler)
        scaled_val = scale_data(val_df, scaler)
        
        X_train, y_train = create_sequences(scaled_train, input_window=window_size)
        X_val, y_val = create_sequences(scaled_val, input_window=window_size)
        
        train_loader = DataLoader(SolarDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(SolarDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
        
        # 2. Instanciar y Entrenar
        model = SolarLSTM(output_window=OUTPUT_WINDOW) # El modelo se adapta al N_FEATURES global
        model, history = train_model(model, train_loader, val_loader, device)
        
        # 3. Evaluar con Ventana Deslizante
        print(f"  Evaluando con Ventana Deslizante...")
        actuals, preds = evaluate_sliding_window(model, scaled_val, scaler, device, window_size)
        
        if actuals is not None:
            metrics = compute_metrics(actuals, preds, name=name)
            results.append(metrics)
            all_predictions[name] = (actuals, preds)
            print(f"  Resultados {name}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")

    # ──────────────────────────────────────────────────────────────────
    # REPORTE FINAL
    # ──────────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("  TABLA COMPARATIVA FINAL")
    print("=" * 70)
    print(df_results.to_string(index=False))
    
    # Guardar resultados
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'grid_search_results.csv'), index=False)
    
    # Gráfica Comparativa (Últimos 3 días de validación para ver detalle)
    plt.figure(figsize=(15, 7))
    first_name = list(all_predictions.keys())[0]
    actuals_sample = all_predictions[first_name][0][-36:] # Últimas 36h
    
    plt.plot(actuals_sample, label='Real', color='black', linewidth=2, linestyle='--')
    
    for name, (act, pre) in all_predictions.items():
        plt.plot(pre[-36:], label=f'Pred {name}', alpha=0.8)
        
    plt.title('Comparación de Predicciones según Ventana de Memoria (Zoom 3 días)', fontsize=14, fontweight='bold')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, 'grid_comparison_plot.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"\n  Gráfica comparativa guardada en: {plot_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
