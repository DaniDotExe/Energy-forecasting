import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import json

# ──────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN Y RUTAS
# ──────────────────────────────────────────────────────────────────────
warnings.filterwarnings('ignore')

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data-hourly.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'EDA', 'SARIMAX_LSTM_Comparison')
MONTHLY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'Mensuales-de-diarios')
os.makedirs(MONTHLY_OUTPUT_DIR, exist_ok=True)

# Hiperparámetros LSTM
INPUT_WINDOW = 168   # 14 días solares
OUTPUT_WINDOW = 12   # 1 día solar
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
MAX_EPOCHS = 100     # Reducido para agilidad, usualmente 300
PATIENCE = 15

# Variables
EXOG_COLS = ['Irradiancia_It', 'Temperatura_Tt', 'Humedad_Ht', 'Viento_Wt']
TIME_COLS = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
TARGET_COL = 'kWh'
ALL_FEATURES = EXOG_COLS + TIME_COLS + [TARGET_COL]
N_FEATURES = len(ALL_FEATURES)

# SARIMAX
SARIMAX_ORDER = (2, 0, 1)
SARIMAX_SEASONAL = (1, 1, 1, 12)

# ──────────────────────────────────────────────────────────────────────
# UTILIDADES DE DATOS
# ──────────────────────────────────────────────────────────────────────
def add_cyclic_features(df):
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df['hour'] = df['Fecha_Hora'].dt.hour
    df['month'] = df['Fecha_Hora'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE (%)': mape(y_true, y_pred)
    }

class SolarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def create_sequences(data, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i : i + input_window])
        y.append(data[i + input_window : i + input_window + output_window, -1])
    return np.array(X), np.array(y)

# ──────────────────────────────────────────────────────────────────────
# MODELO LSTM
# ──────────────────────────────────────────────────────────────────────
class SolarLSTM(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, output_window=OUTPUT_WINDOW):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_window)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)

# ──────────────────────────────────────────────────────────────────────
# PROCESO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  SARIMAX-LSTM ENERGY FORECASTING PIPELINE")
    print("="*70)

    # 1. Carga de datos
    df = pd.read_csv(DATA_PATH)
    df = add_cyclic_features(df)
    
    # Split 60/20/20
    n = len(df)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Escalado
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df[ALL_FEATURES])
    scaled_val = scaler.transform(val_df[ALL_FEATURES])
    scaled_test = scaler.transform(test_df[ALL_FEATURES])

    # 2. Entrenamiento LSTM
    model_path = os.path.join(OUTPUT_DIR, 'best_lstm.pth')
    history_path = os.path.join(OUTPUT_DIR, 'train_history.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SolarLSTM().to(device)
    
    if os.path.exists(model_path) and os.path.exists(history_path):
        print("\n[1/2] Cargando LSTM y su historial ya entrenados...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        print("\n[1/2] Entrenando LSTM...")
        X_train, y_train = create_sequences(scaled_train)
        # Para validación necesitamos contexto de train
        full_scaled = np.concatenate([scaled_train, scaled_val], axis=0)
        X_val, y_val = [], []
        for i in range(n_train, len(full_scaled) - OUTPUT_WINDOW + 1, OUTPUT_WINDOW):
            X_val.append(full_scaled[i-INPUT_WINDOW : i])
            y_val.append(full_scaled[i : i+OUTPUT_WINDOW, -1])
        X_val, y_val = np.array(X_val), np.array(y_val)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        train_loader = DataLoader(SolarDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            t_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                t_losses.append(loss.item())
            
            model.eval()
            v_losses = []
            with torch.no_grad():
                val_in = torch.FloatTensor(X_val).to(device)
                val_out = torch.FloatTensor(y_val).to(device)
                v_pred = model(val_in)
                v_loss = criterion(v_pred, val_out)
                v_losses.append(v_loss.item())
            
            avg_t = np.mean(t_losses)
            avg_v = np.mean(v_losses)
            history['train_loss'].append(avg_t)
            history['val_loss'].append(avg_v)

            if (epoch+1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}: Train Loss={avg_t:.6f}, Val Loss={avg_v:.6f}")

            if avg_v < best_val_loss:
                best_val_loss = avg_v
                torch.save(model.state_dict(), model_path)
                with open(history_path, 'w') as f:
                    json.dump(history, f)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"    Early stopping en época {epoch+1}")
                    break

        # Cargar mejor modelo
        model.load_state_dict(torch.load(model_path))
    
    # Recalcular X_val para el test si no se entrenó (necesario para X_test)
    X_val, y_val = [], []
    full_scaled = np.concatenate([scaled_train, scaled_val], axis=0)
    for i in range(n_train, len(full_scaled) - OUTPUT_WINDOW + 1, OUTPUT_WINDOW):
        X_val.append(full_scaled[i-INPUT_WINDOW : i])
        y_val.append(full_scaled[i : i+OUTPUT_WINDOW, -1])
    X_val, y_val = np.array(X_val), np.array(y_val)
    
    # 3. Entrenamiento SARIMAX
    sarimax_path = os.path.join(OUTPUT_DIR, 'sarimax_results.pkl')
    if os.path.exists(sarimax_path):
        print("\n[2/2] Cargando SARIMAX ya entrenado...")
        with open(sarimax_path, 'rb') as f:
            sarimax_result = pickle.load(f)
    else:
        print("\n[2/2] Entrenando SARIMAX(2,0,1)(1,1,1)12...")
        sarimax_model = SARIMAX(
            train_df[TARGET_COL],
            exog=train_df[EXOG_COLS],
            order=SARIMAX_ORDER,
            seasonal_order=SARIMAX_SEASONAL,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        t0_sarima = time.time()
        sarimax_result = sarimax_model.fit(disp=False, maxiter=50)
        print(f"    SARIMAX entrenado en {time.time() - t0_sarima:.1f}s")
        with open(sarimax_path, 'wb') as f:
            pickle.dump(sarimax_result, f)
    
    # 4. Inferencia en TEST
    print("\nEvaluando en Test...")
    # SARIMAX Test
    sarimax_preds = sarimax_result.get_forecast(steps=len(test_df), exog=test_df[EXOG_COLS]).predicted_mean.values
    
    # LSTM Test (Recursivo o Directo? Usaremos bloques de 12h para coincidir con el pipeline)
    test_full_scaled = np.concatenate([scaled_val, scaled_test], axis=0)
    lstm_test_X = []
    for i in range(len(scaled_val), len(test_full_scaled) - OUTPUT_WINDOW + 1, OUTPUT_WINDOW):
        lstm_test_X.append(test_full_scaled[i-INPUT_WINDOW : i])
    
    lstm_test_X = np.array(lstm_test_X)
    model.eval()
    with torch.no_grad():
        lstm_scaled_preds = model(torch.FloatTensor(lstm_test_X).to(device)).cpu().numpy().flatten()
    
    # Des-escalar LSTM
    dummy = np.zeros((len(lstm_scaled_preds), N_FEATURES))
    dummy[:, -1] = lstm_scaled_preds
    lstm_preds = scaler.inverse_transform(dummy)[:, -1]
    
    # Ajustar longitudes por si acaso
    min_len = min(len(test_df), len(sarimax_preds), len(lstm_preds))
    y_true = test_df[TARGET_COL].values[:min_len]
    sarimax_preds = sarimax_preds[:min_len]
    lstm_preds = lstm_preds[:min_len]
    test_dates = test_df['Fecha_Hora'].iloc[:min_len]

    # Métricas
    metrics_s = compute_metrics(y_true, sarimax_preds)
    metrics_l = compute_metrics(y_true, lstm_preds)

    # ──────────────────────────────────────────────────────────────────────
    # GENERACIÓN DE REPORTES VISUALES
    # ──────────────────────────────────────────────────────────────────────
    print("\nGenerando gráficas de reporte...")

    # 01_lstm_loss.png
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('LSTM Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, '01_lstm_loss.png'), dpi=200, bbox_inches='tight')

    # 02_prediccion_vs_real_full.png (Todo el set de Test - 20%)
    plt.figure(figsize=(16, 6))
    plt.plot(test_dates, y_true, label='Real (2023)', color='black', linewidth=1, alpha=0.8)
    plt.plot(test_dates, sarimax_preds, label='SARIMAX', linestyle='--', alpha=0.6)
    plt.plot(test_dates, lstm_preds, label='LSTM', linestyle='--', alpha=0.6)
    plt.title('Comparación en Set de Test Completo (20% Final - Datos No Vistos)', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, '02_prediccion_vs_real_full.png'), dpi=200, bbox_inches='tight')

    # 07_prediccion_vs_real_zoom.png (Zoom 7 días del test)
    plt.figure(figsize=(14, 6))
    zoom = 24 * 7 # Una semana completa
    plt.plot(test_dates[:zoom], y_true[:zoom], label='Real', color='black', linewidth=1.5)
    plt.plot(test_dates[:zoom], sarimax_preds[:zoom], label='SARIMAX', linestyle='--', alpha=0.8)
    plt.plot(test_dates[:zoom], lstm_preds[:zoom], label='LSTM', linestyle='--', alpha=0.8)
    plt.title('Zoom: Predicción vs Real en Test (Primera Semana)', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, '07_prediccion_vs_real_zoom.png'), dpi=200, bbox_inches='tight')

    # 03_error_por_mes.png
    err_df = pd.DataFrame({'Fecha': test_dates, 'SARIMAX_Err': np.abs(y_true - sarimax_preds), 'LSTM_Err': np.abs(y_true - lstm_preds)})
    err_df['Month'] = err_df['Fecha'].dt.strftime('%Y-%m')
    monthly_err = err_df.groupby('Month').mean()
    
    monthly_err[['SARIMAX_Err', 'LSTM_Err']].plot(kind='bar', figsize=(10, 5), color=['#1f77b4', '#ff7f0e'])
    plt.title('Error Medio Absoluto por Mes (Test)', fontsize=14, fontweight='bold')
    plt.ylabel('MAE (kWh)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, '03_error_por_mes.png'), dpi=200, bbox_inches='tight')

    # 04_residuos_sarimax.png
    residuos = y_true - sarimax_preds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(residuos, color='green', alpha=0.6)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuos SARIMAX en el Tiempo')
    sns.histplot(residuos, kde=True, ax=axes[1], color='purple')
    axes[1].set_title('Distribución de Residuos')
    plt.savefig(os.path.join(OUTPUT_DIR, '04_residuos_sarimax.png'), dpi=200, bbox_inches='tight')

    # 05_serie_completa.png
    plt.figure(figsize=(16, 6))
    plt.plot(df['Fecha_Hora'], df[TARGET_COL], color='gray', alpha=0.3, label='Historial (Train/Val)')
    plt.plot(test_dates, y_true, color='black', label='Real (Test/2023)', linewidth=1)
    plt.plot(test_dates, lstm_preds, color='orange', label='LSTM Pred (Test)', alpha=0.7)
    plt.axvline(x=test_dates.iloc[0], color='red', linestyle='--', label='Punto de Corte Test (20%)')
    plt.title('Serie Completa y Predicciones en Test (20% Final)', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('kWh')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_serie_completa.png'), dpi=200, bbox_inches='tight')

    # 06_tabla_metricas.png
    metrics_df = pd.DataFrame([
        {'Modelo': 'SARIMAX', **metrics_s},
        {'Modelo': 'LSTM', **metrics_l}
    ]).round(2)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')
    tbl = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2)
    plt.title('Tabla de Métricas Comparativas (Horario)', y=1.2, fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, '06_tabla_metricas.png'), dpi=200, bbox_inches='tight')

    # comparacion_final.png (Bar chart RMSE/MAE)
    metrics_melted = metrics_df.melt(id_vars='Modelo', value_vars=['RMSE', 'MAE'])
    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics_melted, x='variable', y='value', hue='Modelo')
    plt.title('Comparación Final de Error (RMSE vs MAE)', fontsize=14, fontweight='bold')
    plt.ylabel('Valor (kWh)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparacion_final.png'), dpi=200, bbox_inches='tight')

    # ──────────────────────────────────────────────────────────────────────
    # EVALUACIÓN MENSUAL (AGREGADA)
    # ──────────────────────────────────────────────────────────────────────
    print("\nEvaluando resultados agregados mensualmente...")
    
    # Agregación de Test
    test_results = pd.DataFrame({
        'Fecha': test_dates,
        'Real': y_true,
        'SARIMAX': sarimax_preds,
        'LSTM': lstm_preds
    })
    # Resamplear por mes sumando valores numéricos (kWh)
    test_monthly = test_results.set_index('Fecha').resample('ME').sum().reset_index()

    # Métricas Mensuales
    m_metrics_s = compute_metrics(test_monthly['Real'].values, test_monthly['SARIMAX'].values)
    m_metrics_l = compute_metrics(test_monthly['Real'].values, test_monthly['LSTM'].values)
    
    m_metrics_df = pd.DataFrame([
        {'Modelo': 'SARIMAX', **m_metrics_s},
        {'Modelo': 'LSTM', **m_metrics_l}
    ])
    m_metrics_df.to_csv(os.path.join(MONTHLY_OUTPUT_DIR, 'monthly_metrics.csv'), index=False)

    # Gráfica Mensual: Real vs Predicción
    plt.figure(figsize=(12, 6))
    plt.plot(test_monthly['Fecha'], test_monthly['Real'], marker='o', label='Real (Mensual)', color='black', linewidth=2)
    plt.plot(test_monthly['Fecha'], test_monthly['SARIMAX'], marker='s', label='SARIMAX', linestyle='--', alpha=0.8)
    plt.plot(test_monthly['Fecha'], test_monthly['LSTM'], marker='^', label='LSTM', linestyle='--', alpha=0.8)
    
    # Texto de métricas (incluyendo MAPE)
    textstr = '\n'.join((
        f'LSTM: RMSE={m_metrics_l["RMSE"]:.2f}, MAE={m_metrics_l["MAE"]:.2f}, MAPE={m_metrics_l["MAPE (%)"]:.2f}%',
        f'SARIMAX: RMSE={m_metrics_s["RMSE"]:.2f}, MAE={m_metrics_s["MAE"]:.2f}, MAPE={m_metrics_s["MAPE (%)"]:.2f}%'
    ))
    plt.gca().text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.title('Comparación de Energía Mensual (Suma kWh)', fontsize=14, fontweight='bold')
    plt.ylabel('Energía Total (kWh)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(MONTHLY_OUTPUT_DIR, '01_comparacion_mensual.png'), dpi=200)

    # Gráfica de Error Mensual (Barra)
    test_monthly['SARIMAX_Error'] = np.abs(test_monthly['Real'] - test_monthly['SARIMAX'])
    test_monthly['LSTM_Error'] = np.abs(test_monthly['Real'] - test_monthly['LSTM'])
    test_monthly['Mes_Año'] = test_monthly['Fecha'].dt.strftime('%m-%Y')
    
    total_err_s = test_monthly['SARIMAX_Error'].sum()
    total_err_l = test_monthly['LSTM_Error'].sum()

    ax = test_monthly.set_index('Mes_Año')[['SARIMAX_Error', 'LSTM_Error']].plot(kind='bar', figsize=(10, 5), color=['#1f77b4', '#ff7f0e'])
    plt.title(f'Error Absoluto Mensual (Suma kWh)\nTotal SARIMAX: {total_err_s:.2f} | Total LSTM: {total_err_l:.2f}', fontsize=12, fontweight='bold')
    plt.ylabel('Error (kWh)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(MONTHLY_OUTPUT_DIR, '02_error_mensual_barras.png'), dpi=200, bbox_inches='tight')

    # Serie Completa Mensual
    df_monthly = df.set_index('Fecha_Hora')[TARGET_COL].resample('ME').sum().reset_index()
    plt.figure(figsize=(16, 6))
    plt.plot(df_monthly['Fecha_Hora'], df_monthly[TARGET_COL], color='gray', alpha=0.3, label='Historial Mensual')
    plt.plot(test_monthly['Fecha'], test_monthly['Real'], color='black', label='Real (Test)', linewidth=2)
    plt.plot(test_monthly['Fecha'], test_monthly['SARIMAX'], color='blue', label='SARIMAX Pred', linestyle='--', alpha=0.7)
    plt.plot(test_monthly['Fecha'], test_monthly['LSTM'], color='orange', label='LSTM Pred', alpha=0.8)
    plt.axvline(x=test_monthly['Fecha'].iloc[0], color='red', linestyle='--', label='Inicio Test')
    plt.title('Historial y Predicciones a Escala Mensual (Ambos Modelos)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.savefig(os.path.join(MONTHLY_OUTPUT_DIR, '03_serie_completa_mensual.png'), dpi=200, bbox_inches='tight')

    # Tabla de métricas en imagen
    m_metrics_df = m_metrics_df.round(2)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')
    tbl = ax.table(cellText=m_metrics_df.values, colLabels=m_metrics_df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2)
    plt.title('Métricas de Error Mensual (Suma de kWh)', y=1.2, fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(MONTHLY_OUTPUT_DIR, '04_tabla_metricas_mensual.png'), dpi=200, bbox_inches='tight')

    print(f"\n  Proceso finalizado. Artefactos horarios en {OUTPUT_DIR}")
    print(f"  Artefactos mensuales en {MONTHLY_OUTPUT_DIR}")

if __name__ == '__main__':
    main()
