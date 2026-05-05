"""
LSTM Solar: Comparación de Estrategias de Validación Temporal
=============================================================
Equivalente metodológico al script Sarimax_strategies.py.

Estrategias:
1. División Simple:    Entrena 1 vez, predice TODO el validation de golpe.
2. Ventana Expansiva:  Entrena 1 vez. Día a día agrega datos reales al historial
                       y predice las próximas 12h. NO re-entrena la red.
3. Ventana Deslizante: Entrena 1 vez. Día a día desliza ventana fija de historial
                       y predice las próximas 12h. NO re-entrena la red.

Horizonte de predicción: 12 horas solares (1 día solar completo).
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# Importamos del módulo de modelo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LSTM_model import (
    get_device, load_data, split_data, fit_scalers, scale_data,
    create_sequences, SolarDataset, SolarLSTM, train_model,
    predict_sequences, inverse_scale_target, compute_metrics,
    INPUT_WINDOW, OUTPUT_WINDOW, BATCH_SIZE, ALL_FEATURES, N_FEATURES
)

# ──────────────────────────────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'data-hourly.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'EDA', 'LSTM_training')
os.makedirs(OUTPUT_DIR, exist_ok=True)

STEP_SIZE = OUTPUT_WINDOW  # 12 horas = 1 día solar

# ──────────────────────────────────────────────────────────────────────
# ESTRATEGIA 1: DIVISIÓN SIMPLE
# ──────────────────────────────────────────────────────────────────────
def strategy_simple_recursive(model, scaled_train, scaled_val, scaler, device):
    """
    PREDICCIÓN RECURSIVA (Estática):
    Predice las primeras 12h usando el final de Train.
    Luego usa esas PREDICCIONES como entrada para las siguientes 12h.
    No usa datos reales de kWh de validación.
    (Nota: Para las exógenas se siguen usando los datos reales de validación
    asumiendo que son 'pronósticos climáticos conocidos').
    """
    print("\n  Ejecutando Estrategia 1: Simple (Recursiva / Estática)...")
    t0 = time.time()
    n_val = len(scaled_val)
    local_scaled_val = scaled_val.copy() # Copia para no afectar otras estrategias
    
    # Empezamos con las últimas 84h de Train
    current_context = scaled_train[-INPUT_WINDOW:].copy()
    
    preds_scaled = []
    actuals_scaled = []

    for i in range(0, n_val - OUTPUT_WINDOW + 1, STEP_SIZE):
        # 1. Preparar secuencia de entrada
        seq_x = current_context[-INPUT_WINDOW:].reshape(1, INPUT_WINDOW, N_FEATURES)
        
        # 2. Predecir
        pred_kwh_scaled = predict_sequences(model, seq_x, device)[0] # (12,)
        preds_scaled.append(pred_kwh_scaled)
        actuals_scaled.append(local_scaled_val[i : i + OUTPUT_WINDOW, -1])

        # 3. Actualizar contexto para la SIGUIENTE predicción
        next_block = local_scaled_val[i : i + OUTPUT_WINDOW].copy()
        next_block[:, -1] = pred_kwh_scaled
        
        current_context = np.concatenate([current_context, next_block], axis=0)

    preds_flat = np.concatenate(preds_scaled)
    actuals_flat = np.concatenate(actuals_scaled)

    preds_kwh = inverse_scale_target(preds_flat, scaler)
    actuals_kwh = inverse_scale_target(actuals_flat, scaler)

    elapsed = time.time() - t0
    m = compute_metrics(actuals_kwh, preds_kwh, "1. Simple (Recursiva)")
    print(f"    RMSE: {m['RMSE']:,.2f} | MAE: {m['MAE']:,.2f} | "
          f"MAPE: {m['MAPE (%)']:.1f}% | Tiempo: {elapsed:.1f}s")
    return m, preds_kwh, actuals_kwh

# ──────────────────────────────────────────────────────────────────────
# ESTRATEGIA 2: VENTANA EXPANSIVA
# ──────────────────────────────────────────────────────────────────────
def strategy_expanding(model, scaled_train, scaled_val, scaler, device, val_ts):
    """
    Entrena 1 sola vez. Día a día:
      1. Construye secuencia con TODO el historial acumulado.
      2. Predice las próximas 12 horas.
      3. Agrega las 12 horas reales al historial.
    NO re-entrena la red en cada paso.
    """
    print("\n  Ejecutando Estrategia 2: Ventana Expansiva...")
    t0 = time.time()
    n_val = len(scaled_val)

    # Historial acumulativo: empieza con todo train
    history = scaled_train.copy()

    preds_scaled = []
    actuals_scaled = []

    for i in range(0, n_val - OUTPUT_WINDOW + 1, STEP_SIZE):
        # Secuencia de entrada: últimas INPUT_WINDOW horas del historial
        if len(history) < INPUT_WINDOW:
            continue
        seq_x = history[-INPUT_WINDOW:]
        seq_x = seq_x.reshape(1, INPUT_WINDOW, N_FEATURES)

        pred = predict_sequences(model, seq_x, device)
        preds_scaled.append(pred[0])
        actuals_scaled.append(scaled_val[i : i + OUTPUT_WINDOW, -1])

        # Expandimos el historial con los datos REALES del día recién observado
        real_block = scaled_val[i : i + OUTPUT_WINDOW]
        history = np.concatenate([history, real_block], axis=0)

    preds_flat = np.concatenate(preds_scaled)
    actuals_flat = np.concatenate(actuals_scaled)

    preds_kwh = inverse_scale_target(preds_flat, scaler)
    actuals_kwh = inverse_scale_target(actuals_flat, scaler)

    elapsed = time.time() - t0
    m = compute_metrics(actuals_kwh, preds_kwh, "2. Ventana Expansiva")
    print(f"    RMSE: {m['RMSE']:,.2f} | MAE: {m['MAE']:,.2f} | "
          f"MAPE: {m['MAPE (%)']:.1f}% | Tiempo: {elapsed:.1f}s")
    return m, preds_kwh, actuals_kwh

# ──────────────────────────────────────────────────────────────────────
# ESTRATEGIA 3: VENTANA DESLIZANTE
# ──────────────────────────────────────────────────────────────────────
def strategy_sliding(model, scaled_train, scaled_val, scaler, device, val_ts):
    """
    Entrena 1 sola vez. Día a día:
      1. Toma una ventana FIJA del historial (tamaño = len(train)).
      2. Construye secuencia con las últimas INPUT_WINDOW horas de esa ventana.
      3. Predice las próximas 12 horas.
      4. Desliza la ventana: agrega datos reales, descarta los más antiguos.
    NO re-entrena la red en cada paso.
    """
    print("\n  Ejecutando Estrategia 3: Ventana Deslizante...")
    t0 = time.time()
    n_val = len(scaled_val)
    n_train = len(scaled_train)

    # Dataset completo para poder deslizar
    full_scaled = np.concatenate([scaled_train, scaled_val], axis=0)

    preds_scaled = []
    actuals_scaled = []

    for i in range(0, n_val - OUTPUT_WINDOW + 1, STEP_SIZE):
        # Ventana fija: siempre n_train registros
        global_end = n_train + i
        window_start = global_end - n_train

        window = full_scaled[window_start : global_end]

        # Secuencia de entrada: últimas INPUT_WINDOW horas de la ventana
        seq_x = window[-INPUT_WINDOW:]
        seq_x = seq_x.reshape(1, INPUT_WINDOW, N_FEATURES)

        pred = predict_sequences(model, seq_x, device)
        preds_scaled.append(pred[0])
        actuals_scaled.append(scaled_val[i : i + OUTPUT_WINDOW, -1])

    preds_flat = np.concatenate(preds_scaled)
    actuals_flat = np.concatenate(actuals_scaled)

    preds_kwh = inverse_scale_target(preds_flat, scaler)
    actuals_kwh = inverse_scale_target(actuals_flat, scaler)

    elapsed = time.time() - t0
    m = compute_metrics(actuals_kwh, preds_kwh, "3. Ventana Deslizante")
    print(f"    RMSE: {m['RMSE']:,.2f} | MAE: {m['MAE']:,.2f} | "
          f"MAPE: {m['MAPE (%)']:.1f}% | Tiempo: {elapsed:.1f}s")
    return m, preds_kwh, actuals_kwh

# ──────────────────────────────────────────────────────────────────────
# GRÁFICAS
# ──────────────────────────────────────────────────────────────────────
def plot_training_curves(history, output_dir):
    """Curvas de loss de entrenamiento y validación."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['train_loss'], label='Train Loss', color='#2196F3', linewidth=1.5)
    ax.plot(history['val_loss'], label='Validation Loss', color='#FF5722', linewidth=1.5)
    ax.set_xlabel('Época')
    ax.set_ylabel('MSE Loss')
    ax.set_title('LSTM - Curvas de Entrenamiento', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_training_curves.png'), dpi=200)
    plt.close()

def plot_metrics_comparison(df_metrics, output_dir):
    """Gráfico de barras comparativo de RMSE, MAE, MAPE."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['RMSE', 'MAE', 'MAPE (%)']
    colors = ['#1976D2', '#FF9800', '#4CAF50']

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(df_metrics['Estrategia'], df_metrics[metric],
                      color=color, edgecolor='black', alpha=0.85)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor')
        ax.tick_params(axis='x', rotation=15)
        for i, v in enumerate(df_metrics[metric]):
            ax.text(i, v, f'{v:,.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.suptitle('LSTM - Comparación de Estrategias', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_estrategias_metricas.png'), dpi=200, bbox_inches='tight')
    plt.close()

def plot_predictions(actuals_dict, preds_dict, val_dates_aligned, output_dir):
    """Real vs Predicción para cada estrategia + zoom último mes."""
    strategies = list(preds_dict.keys())
    n_strats = len(strategies)

    # --- Real vs Pred por estrategia ---
    fig, axes = plt.subplots(n_strats, 1, figsize=(16, 4 * n_strats), sharex=True)
    if n_strats == 1:
        axes = [axes]
    colors = ['#1976D2', '#FF9800', '#4CAF50']

    for ax, strat, color in zip(axes, strategies, colors):
        n = len(preds_dict[strat])
        dates = val_dates_aligned[:n]
        ax.plot(dates, actuals_dict[strat], label='Real', color='black', alpha=0.6, linewidth=0.8)
        ax.plot(dates, preds_dict[strat], label=f'Pred ({strat})', color=color, alpha=0.8, linewidth=0.8)
        ax.set_ylabel('kWh')
        ax.set_title(f'Real vs Predicción - {strat}', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_real_vs_pred.png'), dpi=200)
    plt.close()

    # --- Error temporal ---
    fig, axes = plt.subplots(n_strats, 1, figsize=(16, 3 * n_strats), sharex=True)
    if n_strats == 1:
        axes = [axes]

    for ax, strat, color in zip(axes, strategies, colors):
        n = len(preds_dict[strat])
        dates = val_dates_aligned[:n]
        error = actuals_dict[strat] - preds_dict[strat]
        ax.plot(dates, error, color=color, alpha=0.6, linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel('Error (kWh)')
        ax.set_title(f'Error Temporal - {strat}', fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_error_temporal.png'), dpi=200)
    plt.close()

    # --- Zoom último mes ---
    last_month = slice(-360, None)
    fig, ax = plt.subplots(figsize=(16, 6))
    ref_strat = strategies[0]
    n_ref = len(preds_dict[ref_strat])
    dates = val_dates_aligned[:n_ref]

    ax.plot(dates.iloc[last_month], actuals_dict[ref_strat][last_month],
            label='Real', color='black', alpha=0.7, linewidth=1.5)
    for strat, color in zip(strategies, colors):
        ax.plot(dates.iloc[last_month], preds_dict[strat][last_month],
                label=strat, color=color, alpha=0.8, linewidth=1)

    ax.set_title('LSTM - Zoom Último Mes de Validación', fontsize=14, fontweight='bold')
    ax.set_ylabel('kWh')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_zoom_ultimo_mes.png'), dpi=200)
    plt.close()

    # --- Comparación temporal 3 estrategias ---
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, actuals_dict[ref_strat],
            label='Real', color='black', alpha=0.5, linewidth=0.7)
    for strat, color in zip(strategies, colors):
        ax.plot(dates, preds_dict[strat],
                label=strat, color=color, alpha=0.7, linewidth=0.7)

    ax.set_title('LSTM - Comparación Temporal de Estrategias', fontsize=14, fontweight='bold')
    ax.set_ylabel('kWh')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_comparacion_temporal.png'), dpi=200)
    plt.close()

# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  LSTM - COMPARACIÓN DE ESTRATEGIAS DE VALIDACIÓN TEMPORAL")
    print("=" * 70)

    # --- Dispositivo ---
    device = get_device()

    # --- Datos ---
    print("-" * 70)
    print("  CARGA DE DATOS")
    print("-" * 70)
    df = load_data(DATA_PATH)
    train_df, val_df, test_df = split_data(df)

    # --- Escalado ---
    print("\n" + "-" * 70)
    print("  ESCALADO Y SECUENCIAS")
    print("-" * 70)
    scaler = fit_scalers(train_df)
    scaled_train = scale_data(train_df, scaler)
    scaled_val = scale_data(val_df, scaler)

    print(f"  Datos escalados: Train {scaled_train.shape}, Val {scaled_val.shape}")

    # --- Secuencias para entrenamiento ---
    # Usamos SOLO datos de train para crear secuencias de entrenamiento
    X_train, y_train = create_sequences(scaled_train)
    print(f"  Secuencias Train: X={X_train.shape}, y={y_train.shape}")

    # Secuencias de validación (para el loss durante entrenamiento)
    # Necesitamos incluir el final de train como contexto
    train_val_scaled = np.concatenate([scaled_train, scaled_val], axis=0)
    n_train = len(scaled_train)

    # Secuencias que EMPIEZAN a predecir dentro de validation
    X_val_seq, y_val_seq = [], []
    for i in range(n_train, len(train_val_scaled) - OUTPUT_WINDOW + 1):
        start = i - INPUT_WINDOW
        if start < 0:
            continue
        X_val_seq.append(train_val_scaled[start : i])
        y_val_seq.append(train_val_scaled[i : i + OUTPUT_WINDOW, -1])

    X_val_seq = np.array(X_val_seq)
    y_val_seq = np.array(y_val_seq)
    print(f"  Secuencias Val:   X={X_val_seq.shape}, y={y_val_seq.shape}")

    # --- DataLoaders ---
    train_ds = SolarDataset(X_train, y_train)
    val_ds = SolarDataset(X_val_seq, y_val_seq)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # --- Modelo ---
    print("\n" + "-" * 70)
    print("  ENTRENAMIENTO LSTM")
    print("-" * 70)
    model = SolarLSTM()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Arquitectura: LSTM(features={N_FEATURES}, hidden={model.hidden_size}, "
          f"layers={model.num_layers})")
    print(f"  Parámetros totales: {total_params:,}")

    model, history = train_model(model, train_loader, val_loader, device)

    # --- Curvas de entrenamiento ---
    plot_training_curves(history, OUTPUT_DIR)
    print(f"  Curvas guardadas en: {OUTPUT_DIR}")

    # --- Estrategias de validación ---
    print("\n" + "=" * 70)
    print("  ESTRATEGIAS DE VALIDACIÓN TEMPORAL")
    print("=" * 70)

    val_ts = val_df['kWh'].values

    m1, pred1, act1 = strategy_simple_recursive(model, scaled_train, scaled_val, scaler, device)
    m2, pred2, act2 = strategy_expanding(model, scaled_train, scaled_val, scaler, device, val_ts)
    m3, pred3, act3 = strategy_sliding(model, scaled_train, scaled_val, scaler, device, val_ts)

    # --- Tabla comparativa ---
    results = [m1, m2, m3]
    df_metrics = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("  RESULTADOS LSTM")
    print("=" * 70)
    print(df_metrics.to_string(index=False))

    # --- Exportación CSV ---
    csv_path = os.path.join(OUTPUT_DIR, 'lstm_comparacion_estrategias.csv')
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n  Métricas guardadas en: {csv_path}")

    # Predicciones CSV
    n_common = min(len(pred1), len(pred2), len(pred3))
    df_preds = pd.DataFrame({
        'Real': act1[:n_common],
        'Simple': pred1[:n_common],
        'Expansiva': pred2[:n_common],
        'Deslizante': pred3[:n_common]
    })
    preds_csv = os.path.join(OUTPUT_DIR, 'lstm_predicciones.csv')
    df_preds.to_csv(preds_csv, index=False)
    print(f"  Predicciones guardadas en: {preds_csv}")

    # --- Gráficas ---
    print("\n  Generando gráficas...")
    plot_metrics_comparison(df_metrics, OUTPUT_DIR)

    # Fechas alineadas a las predicciones (cada STEP_SIZE desde el inicio de val)
    # Las predicciones son bloques de 12, tomamos las fechas correspondientes
    n_pred = len(pred1)
    # Generamos índices de las horas predichas dentro del validation
    pred_indices = []
    n_val = len(scaled_val)
    for i in range(0, n_val - OUTPUT_WINDOW + 1, STEP_SIZE):
        for j in range(OUTPUT_WINDOW):
            pred_indices.append(i + j)
            if len(pred_indices) >= n_pred:
                break
        if len(pred_indices) >= n_pred:
            break
    val_dates_aligned = val_df['Fecha_Hora'].iloc[pred_indices[:n_pred]].reset_index(drop=True)

    preds_dict = {'Simple': pred1[:n_common], 'Expansiva': pred2[:n_common], 'Deslizante': pred3[:n_common]}
    actuals_dict = {'Simple': act1[:n_common], 'Expansiva': act2[:n_common], 'Deslizante': act3[:n_common]}

    plot_predictions(actuals_dict, preds_dict, val_dates_aligned[:n_common], OUTPUT_DIR)

    print(f"\n  Todas las gráficas guardadas en: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("  PROCESO COMPLETADO")
    print("=" * 70)

if __name__ == '__main__':
    main()
