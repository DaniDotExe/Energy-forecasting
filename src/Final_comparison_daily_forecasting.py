import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_HOURLY_PATH = os.path.join(BASE_DIR, '..', 'data', 'data-hourly.csv')
DATA_MONTHLY_PATH = os.path.join(BASE_DIR, '..', 'data', 'monthly-kwh.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'EDA', 'FINAL_COMPARISON')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_WINDOW = 168  # 14 días para LSTM/MLP/XGB_Mem
EXOG_COLS = ['Irradiancia_It', 'Temperatura_Tt', 'Humedad_Ht', 'Viento_Wt']
FEATURES = EXOG_COLS + ['kWh']

# Importamos arquitecturas
sys.path.insert(0, BASE_DIR)
from LSTM_model import SolarLSTM, get_device, add_cyclic_features
from MLP_model import SolarMLP

# ──────────────────────────────────────────────────────────────────────
# PREPARACIÓN DE DATOS
# ──────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_HOURLY_PATH)
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df = add_cyclic_features(df)
    
    # Split 90/10
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df, df

# ──────────────────────────────────────────────────────────────────────
# MODELOS
# ──────────────────────────────────────────────────────────────────────

def run_sarimax(train_df, test_df):
    print("\n[1/5] Ejecutando SARIMAX...")
    model = SARIMAX(train_df['kWh'], exog=train_df[EXOG_COLS], 
                    order=(2,0,1), seasonal_order=(1,1,1,12))
    fit = model.fit(disp=False)
    preds = fit.get_forecast(steps=len(test_df), exog=test_df[EXOG_COLS]).predicted_mean
    return np.maximum(preds.values, 0)

def train_nn(model, train_scaled, device, epochs=20):
    X, y = [], []
    for i in range(len(train_scaled) - INPUT_WINDOW):
        X.append(train_scaled[i : i+INPUT_WINDOW])
        y.append(train_scaled[i+INPUT_WINDOW, -1])
    
    X, y = torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    loader = DataLoader(list(zip(X, y)), batch_size=64, shuffle=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for e in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(), by)
            loss.backward()
            optimizer.step()
    return model

def predict_nn_recursive(model, initial_context, steps, test_scaled, train_scaled, device):
    model.eval()
    curr = initial_context.copy()
    preds = []
    with torch.no_grad():
        for i in range(steps):
            inp = torch.FloatTensor(curr[-INPUT_WINDOW:]).unsqueeze(0).to(device)
            p = model(inp).item()
            preds.append(p)
            
            # Inyectar exógenas reales del futuro
            f_idx = len(train_scaled) + i
            if f_idx < len(test_scaled):
                next_row = test_scaled[f_idx].copy()
                next_row[-1] = p
                curr = np.vstack([curr, next_row])
    return np.array(preds)

# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("  COMPARACIÓN DIARIA FINAL: 5 MODELOS EN COMPETENCIA")
    print("="*70)
    
    train_df, test_df, full_df = load_data()
    device = get_device()
    
    # 1. SARIMAX
    sarima_preds = run_sarimax(train_df, test_df)
    
    # Prepara escalas para Redes y XGB
    feat_cols = EXOG_COLS + ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'kWh']
    scaler = MinMaxScaler()
    s_train = scaler.fit_transform(train_df[feat_cols])
    s_full = scaler.transform(full_df[feat_cols])
    n_feat = len(feat_cols)
    
    # 2. LSTM
    print("[2/5] Entrenando LSTM...")
    lstm = SolarLSTM(n_features=n_feat, hidden_size=64, num_layers=2, output_window=1)
    lstm = train_nn(lstm, s_train, device, epochs=15)
    lstm_scaled = predict_nn_recursive(lstm, s_train, len(test_df), s_full, s_train, device)
    
    # 3. MLP
    print("[3/5] Entrenando MLP Complejo...")
    mlp = SolarMLP(input_size=INPUT_WINDOW * n_feat, hidden_size=256)
    mlp = train_nn(mlp, s_train, device, epochs=15)
    mlp_scaled = predict_nn_recursive(mlp, s_train, len(test_df), s_full, s_train, device)
    
    # 4. XGBoost Memoria
    print("[4/5] Entrenando XGBoost (Memoria)...")
    X_xgb, y_xgb = [], []
    for i in range(len(s_train)-INPUT_WINDOW):
        X_xgb.append(s_train[i:i+INPUT_WINDOW].flatten())
        y_xgb.append(s_train[i+INPUT_WINDOW, -1])
    xgb_mem = xgb.XGBRegressor(n_estimators=100, max_depth=6).fit(np.array(X_xgb), np.array(y_xgb))
    
    # Predicción recursiva XGB Mem
    curr_xgb = s_train.copy()
    xgb_mem_scaled = []
    for i in range(len(test_df)):
        inp = curr_xgb[-INPUT_WINDOW:].flatten().reshape(1, -1)
        p = xgb_mem.predict(inp)[0]
        xgb_mem_scaled.append(p)
        f_idx = len(s_train) + i
        if f_idx < len(s_full):
            next_row = s_full[f_idx].copy()
            next_row[-1] = p
            curr_xgb = np.vstack([curr_xgb, next_row])
            
    # 5. XGBoost Directo
    print("[5/5] Entrenando XGBoost Directo (Solo Clima)...")
    xgb_dir = xgb.XGBRegressor(n_estimators=100).fit(s_train[:, :-1], s_train[:, -1])
    xgb_dir_scaled = xgb_dir.predict(s_full[len(s_train):, :-1])
    
    # Des-escalar todos
    def denorm(p_scaled):
        dummy = np.zeros((len(p_scaled), n_feat))
        dummy[:, -1] = p_scaled
        return scaler.inverse_transform(dummy)[:, -1]
    
    lstm_f = denorm(lstm_scaled)
    mlp_f = denorm(mlp_scaled)
    xgb_mem_f = denorm(np.array(xgb_mem_scaled))
    xgb_dir_f = denorm(xgb_dir_scaled)
    
    # Consolidar resultados diarios para comparar con monthly-kwh
    test_res = test_df.copy()
    test_res['SARIMAX'] = sarima_preds
    test_res['LSTM'] = lstm_f
    test_res['MLP'] = mlp_f
    test_res['XGB_Mem'] = xgb_mem_f
    test_res['XGB_Direct'] = xgb_dir_f
    
    # Agrupar por mes para la gráfica final comparativa
    monthly_comparison = test_res.groupby(test_res['Fecha_Hora'].dt.to_period('M')).agg({
        'kWh': 'sum',
        'SARIMAX': 'sum',
        'LSTM': 'sum',
        'MLP': 'sum',
        'XGB_Mem': 'sum',
        'XGB_Direct': 'sum'
    }).reset_index()
    monthly_comparison['Mes'] = monthly_comparison['Fecha_Hora'].astype(str)
    
    # --- Métricas y Gráfica ---
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = []
    models = ['SARIMAX', 'LSTM', 'MLP', 'XGB_Mem', 'XGB_Direct']
    
    for m in models:
        y_true = monthly_comparison['kWh']
        y_pred = monthly_comparison[m]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape_val = calculate_mape(y_true, y_pred)
        metrics.append({
            'Modelo': m,
            'RMSE_Mensual': rmse,
            'MAE_Mensual': mae,
            'MAPE_Mensual (%)': mape_val
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(OUTPUT_DIR, 'daily_pipeline_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # Identificar ganador
    winner = metrics_df.loc[metrics_df['MAPE_Mensual (%)'].idxmin()]
    
    print("\n" + "-"*70)
    print("  RESULTADOS MENSUALIZADOS (Acumulado escala diaria)")
    print("-"*70)
    print(monthly_comparison[['Mes', 'kWh', 'SARIMAX', 'XGB_Direct']].to_string(index=False))
    
    print("\n" + "-"*70)
    print("  TABLA DE MÉTRICAS FINALES")
    print("-"*70)
    print(metrics_df.to_string(index=False))
    print(f"\nGANADOR: {winner['Modelo']} con {winner['MAPE_Mensual (%)']:.2f}% de error MAPE.")
    print("-" * 70)

    # Gráfica Mejorada
    plt.figure(figsize=(14, 7))
    # Línea Real destacada (Negra, más gruesa)
    plt.plot(monthly_comparison['Mes'], monthly_comparison['kWh'], 
             marker='o', label='REAL (Generación Real)', color='black', linewidth=4, zorder=5)
    
    # Líneas de modelos (Más delgadas y punteadas)
    styles = {'SARIMAX': 's--', 'LSTM': '^--', 'MLP': 'x--', 'XGB_Mem': 'd--', 'XGB_Direct': 'p-'}
    colors = {'SARIMAX': 'blue', 'LSTM': 'green', 'MLP': 'orange', 'XGB_Mem': 'purple', 'XGB_Direct': 'red'}
    
    for m in models:
        plt.plot(monthly_comparison['Mes'], monthly_comparison[m], 
                 styles[m], label=m, color=colors[m], alpha=0.8)
    
    plt.title('Batalla Final: Generación Real vs 5 Modelos (Escala Diaria)', fontsize=15, fontweight='bold')
    plt.ylabel('kWh Mensual Acumulado', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, 'daily_comparison_final.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"\n  Resultados guardados en: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
