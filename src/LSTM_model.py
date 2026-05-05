"""
LSTM Solar Energy Forecasting - Modelo y Entrenamiento
======================================================
Módulo con la arquitectura LSTM, dataset, entrenamiento y utilidades.
Diseñado para GPU CUDA. Equivalente metodológico al SARIMAX.

Variables exógenas: Irradiancia, Temperatura, Humedad, Viento
Variable objetivo: kWh
Ventana entrada: 84h (7 días solares)
Ventana salida:  12h (1 día solar)
"""

import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ──────────────────────────────────────────────────────────────────────
# REPRODUCIBILIDAD
# ──────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN MEJORADA
# ──────────────────────────────────────────────────────────────────────
INPUT_WINDOW = 168   # 14 días solares
OUTPUT_WINDOW = 12  
HIDDEN_SIZE = 128    
NUM_LAYERS = 3       
DROPOUT = 0.3        
LEARNING_RATE = 5e-4 
BATCH_SIZE = 32      
MAX_EPOCHS = 300
PATIENCE = 25        
LR_PATIENCE = 10

EXOG_COLS = ['Irradiancia_It', 'Temperatura_Tt', 'Humedad_Ht', 'Viento_Wt']
TIME_COLS = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
TARGET_COL = 'kWh'
ALL_FEATURES = EXOG_COLS + TIME_COLS + [TARGET_COL] 
N_FEATURES = len(ALL_FEATURES)

# ──────────────────────────────────────────────────────────────────────
# DISPOSITIVO
# ──────────────────────────────────────────────────────────────────────
def get_device():
    """Detecta GPU CUDA automáticamente."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("  GPU no disponible, usando CPU")
    print(f"  Dispositivo seleccionado: {dev}\n")
    return dev

# ──────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ──────────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred, name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return {'Estrategia': name, 'RMSE': rmse, 'MAE': mae_val, 'MAPE (%)': mape_val}
# ──────────────────────────────────────────────────────────────────────
# INGENIERÍA DE CARACTERÍSTICAS
# ──────────────────────────────────────────────────────────────────────
def add_cyclic_features(df):
    """Añade codificación cíclica para Hora y Mes."""
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df['hour'] = df['Fecha_Hora'].dt.hour
    df['month'] = df['Fecha_Hora'].dt.month
    
    # Hora (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Mes (1-12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

# ──────────────────────────────────────────────────────────────────────
# CARGA Y PREPROCESAMIENTO
# ──────────────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------
def load_data(data_path):
    """Carga CSV, ordena por fecha, retorna DataFrame."""
    df = pd.read_csv(data_path)
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df = df.sort_values('Fecha_Hora').reset_index(drop=True)
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df = add_cyclic_features(df)
    print(f"  Dataset cargado: {len(df)} registros")
    print(f"  Rango: {df['Fecha_Hora'].iloc[0]} -> {df['Fecha_Hora'].iloc[-1]}")
    return df

def split_data(df):
    """Division temporal 70/20/10 sin shuffle."""
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    print(f"  Train: {len(train)} ({train['Fecha_Hora'].iloc[0]} -> {train['Fecha_Hora'].iloc[-1]})")
    print(f"  Valid: {len(val)} ({val['Fecha_Hora'].iloc[0]} -> {val['Fecha_Hora'].iloc[-1]})")
    print(f"  Test:  {len(test)} ({test['Fecha_Hora'].iloc[0]} -> {test['Fecha_Hora'].iloc[-1]})")
    return train, val, test

def fit_scalers(train_df):
    """Ajusta MinMaxScaler SOLO con datos de entrenamiento."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[ALL_FEATURES].values)
    return scaler

def scale_data(df, scaler):
    """Escala las features usando el scaler pre-ajustado."""
    scaled = scaler.transform(df[ALL_FEATURES].values)
    return scaled  # shape: (N, 5)

# ──────────────────────────────────────────────────────────────────────
# SECUENCIAS TEMPORALES
# ──────────────────────────────────────────────────────────────────────
def create_sequences(data, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW):
    """
    Crea pares (X, y) para entrenamiento supervisado.
    X: (input_window, n_features) - todas las features
    y: (output_window,)           - solo kWh (última columna)
    """
    X, y = [], []
    target_idx = -1  # kWh es la última columna en ALL_FEATURES
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i : i + input_window])
        y.append(data[i + input_window : i + input_window + output_window, target_idx])
    return np.array(X), np.array(y)

# ──────────────────────────────────────────────────────────────────────
# DATASET PYTORCH
# ──────────────────────────────────────────────────────────────────────
class SolarDataset(Dataset):
    """Dataset PyTorch para secuencias temporales solares."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ──────────────────────────────────────────────────────────────────────
# MODELO LSTM
# ──────────────────────────────────────────────────────────────────────
class SolarLSTM(nn.Module):
    """
    LSTM Bidireccional con LayerNorm y capas densas.
    """
    def __init__(self, n_features=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT,
                 output_window=OUTPUT_WINDOW):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Bidireccional
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # LayerNorm (hidden_size * 2 porque es bidireccional)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
        # Reducción de dimensión
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_window)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)
        
        # Tomamos el último estado temporal
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size * 2)
        
        out = self.ln(last_hidden)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)                # (batch, output_window)
        return out

# ──────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device,
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
                lr=LEARNING_RATE, lr_patience=LR_PATIENCE):
    """
    Entrena el modelo LSTM con EarlyStopping y ReduceLROnPlateau.
    Retorna: modelo entrenado, historial de losses.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=lr_patience
    )

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"  Entrenamiento: max {max_epochs} épocas, EarlyStopping={patience}")
    print(f"  Batches train: {len(train_loader)}, val: {len(val_loader)}")
    t0 = time.time()

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        scheduler.step(avg_val)

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Época {epoch+1:3d}/{max_epochs} | "
                  f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {current_lr:.1e}")

        # EarlyStopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    EarlyStopping en época {epoch+1} (mejor val_loss: {best_val_loss:.6f})")
                break

    elapsed = time.time() - t0
    print(f"  Entrenamiento finalizado en {elapsed:.1f}s ({epoch+1} épocas)")

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"  Memoria GPU máxima usada: {mem:.1f} MB")

    model.load_state_dict(best_model_state)
    return model, history

# ──────────────────────────────────────────────────────────────────────
# INFERENCIA
# ──────────────────────────────────────────────────────────────────────
def predict_sequences(model, X, device, batch_size=256):
    """
    Genera predicciones para un array de secuencias.
    X: np.array (N, input_window, n_features)
    Retorna: np.array (N, output_window)
    """
    model.eval()
    dataset = torch.FloatTensor(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)

def inverse_scale_target(scaled_values, scaler):
    """
    Des-escala solo la columna kWh.
    scaled_values: array 1D de valores escalados de kWh.
    """
    target_idx = len(ALL_FEATURES) - 1  # kWh es la última
    dummy = np.zeros((len(scaled_values), len(ALL_FEATURES)))
    dummy[:, target_idx] = scaled_values
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx]
