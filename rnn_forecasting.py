#!/usr/bin/env python3
"""
rnn_forecasting.py

Red Neuronal Recurrente (LSTM Avanzada) en PyTorch para la predicción de generación solar
utilizando la técnica de División Cronológica Simple (70/20/10).
Procesa tanto datos diarios como mensuales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Semilla para reproducibilidad
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# CONFIGURACIÓN GLOBAL Y DISPOSITIVO
# ============================================================
BASE_DIR = r"d:\Software\Energy-forecasting"
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "resultados_rnn")

# Parámetros del modelo
EPOCHS = 1000
BATCH_SIZE = 8
PATIENCE = 100

# Features y target
FEATURE_COLS = ['Mes', 'Temperatura_Tt', 'Viento_Wt', 'Irradiancia_It']
TARGET_COL = 'Total_Generacion'

# Detección de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizando dispositivo: {device}")


# ============================================================
# DEFINICIÓN DEL MODELO LSTM AVANZADO
# ============================================================
class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1):
        super(AdvancedLSTMModel, self).__init__()
        
        # 1. LSTMs Bidireccionales para mejor extracción de contexto histórico
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Como es bidireccional, la entrada a lstm2 es el doble (hidden_size1 * 2)
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # 2. Layer Normalization (hidden_size2 * 2 por ser bidireccional)
        self.layer_norm = nn.LayerNorm(hidden_size2 * 2)
        
        # 3. Capas densas ajustadas
        self.fc1 = nn.Linear(hidden_size2 * 2, 32)
        # 4. LeakyReLU para evitar pérdida de gradientes
        self.act = nn.LeakyReLU(negative_slope=0.01) 
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # Secuencia a través de las LSTMs
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # Mejora: en lugar de tomar solo el último paso (out[:, -1, :]), 
        # promediamos el contexto de toda la ventana temporal (Global Average Pooling)
        out = torch.mean(out, dim=1) 
        
        # Normalización y clasificación/regresión
        out = self.layer_norm(out)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        
        return out


# ============================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================

def create_sequences(X_scaled, y_scaled, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X_scaled)):
        Xs.append(X_scaled[i - lookback:i])
        ys.append(y_scaled[i])
    return np.array(Xs), np.array(ys)


def train_model(model, train_loader, val_loader, epochs, patience):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_val_pred = model(X_val)
                val_loss += criterion(y_val_pred, y_val).item()
        
        if len(val_loader) > 0: val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Early Stopping en época {epoch+1}]")
                break
                
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


def run_pipeline(data_path, output_subdir, lookback, title_prefix):
    """Ejecuta todo el pipeline para un dataset específico."""
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

    # 2. Estandarizar Datos
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_X.fit_transform(df[FEATURE_COLS].values)
    y_scaled = scaler_y.fit_transform(df[[TARGET_COL]].values)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback)
    fechas_seq = df['Fecha'].values[lookback:]

    # 3. Split 70/20/10
    n = len(X_seq)
    train_split = int(n * 0.70)
    val_split = int(n * 0.90)

    X_train, y_train = X_seq[:train_split], y_seq[:train_split]
    X_val, y_val = X_seq[train_split:val_split], y_seq[train_split:val_split]
    X_test, y_test = X_seq[val_split:], y_seq[val_split:]
    fechas_test = fechas_seq[val_split:]

    # 4. DataLoaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)

    # 5. Modelo y Entrenamiento
    model = AdvancedLSTMModel(input_size=len(FEATURE_COLS))
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, PATIENCE)

    # 6. Predicción en todos los sets
    model.eval()
    with torch.no_grad():
        y_train_pred_scaled = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
        y_val_pred_scaled = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
        y_test_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    
    y_train_real = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_val_real = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Cálculo de métricas
    epsilon = 1e-10
    mape = np.mean(np.abs((y_test_real - y_test_pred) / (y_test_real + epsilon))) * 100
    mae = mean_absolute_error(y_test_real, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_test_pred))

    # 7. Graficar
    # --- Gráfica de Loss ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Loss Entrenamiento')
    ax.plot(val_losses, label='Loss Validación')
    ax.set_title(f'Pérdida por Época ({title_prefix}) - Final MAPE: {mape:.2f}%')
    ax.set_xlabel('Época')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'loss_history.png'), dpi=300)
    plt.close()

    # --- Gráfica completa ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(fechas_seq[:train_split], y_train_real, 'b-', label='Real Train', alpha=0.3)
    ax.plot(fechas_seq[train_split:val_split], y_val_real, 'g-', label='Real Val', alpha=0.3)
    ax.plot(fechas_seq[val_split:], y_test_real, 'k-', label='Real Test', alpha=0.3)
    ax.plot(fechas_seq[:train_split], y_train_pred, 'r--', label='Pred Train', alpha=0.8)
    ax.plot(fechas_seq[train_split:val_split], y_val_pred, 'm--', label='Pred Val', alpha=0.8)
    ax.plot(fechas_seq[val_split:], y_test_pred, 'c--', label='Pred Test', alpha=0.8)
    ax.set_title(f'Predicción Completa ({title_prefix}) - MAPE Test: {mape:.2f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'prediccion_completa.png'), dpi=300)
    plt.close()

    # --- Gráfica Zoom Test ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fechas_seq[val_split:], y_test_real, 'b-o', label='Real Test', markersize=4, alpha=0.7)
    ax.plot(fechas_seq[val_split:], y_test_pred, 'r--s', label='Predicción Test', markersize=4, alpha=0.8)
    ax.set_title(f'Predicción Test RNN ({title_prefix}) - MAPE: {mape:.2f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'prediccion_test.png'), dpi=300)
    plt.close()

    print(f"  Resultados Test ({output_subdir}):")
    print(f"  - MAPE: {mape:.2f}%")
    print(f"  - MAE:  {mae:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    
    return mape


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    experimentos = [
        {
            'path': os.path.join(BASE_DIR, "monthly_data.csv"),
            'subdir': 'monthly',
            'lookback': 6,
            'name': 'Datos Mensuales'
        },
        {
            'path': os.path.join(BASE_DIR, "daily_data.csv"),
            'subdir': 'daily',
            'lookback': 30,
            'name': 'Datos Diarios'
        }
    ]

    for exp in experimentos:
        try:
            run_pipeline(exp['path'], exp['subdir'], exp['lookback'], exp['name'])
        except Exception as e:
            print(f"Error procesando {exp['name']}: {e}")

    print("\n=== Proceso completo finalizado. Revisa la carpeta 'resultados_rnn' ===")
