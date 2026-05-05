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
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'monthly-data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'EDA', 'MONTHLY_COMPARISON')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_WINDOW = 12  # Usar 1 año para predecir el siguiente mes
OUTPUT_WINDOW = 1  # Predecir 1 mes a la vez
BATCH_SIZE = 4     # Batch pequeño por pocos datos
EPOCHS = 200

# Importamos MLP
sys.path.insert(0, BASE_DIR)
from MLP_model import SolarMLP
# Reusamos la lógica de LSTM pero ligera
from LSTM_model import SolarLSTM, get_device

# ──────────────────────────────────────────────────────────────────────
# PREPARACIÓN DE DATOS
# ──────────────────────────────────────────────────────────────────────
def prepare_monthly_data():
    df = pd.read_csv(DATA_PATH)
    # Features: Todo menos Mes y kWh_Total
    features = [c for c in df.columns if c not in ['Mes', 'kWh_Total']] + ['kWh_Total']
    
    # Split: Últimos 6 meses para Test
    train_df = df.iloc[:-6].copy()
    test_df = df.iloc[-6:].copy()
    
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df[features])
    scaled_test = scaler.transform(df[features]) # Usamos df completo para tener historial previo al test
    
    return train_df, test_df, scaled_train, scaled_test, scaler, features

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i+window])
        y.append(data[i+window, -1]) # Objetivo: kWh_Total
    return np.array(X), np.array(y)

class MonthlyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ──────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO GENÉRICO (Para LSTM y MLP)
# ──────────────────────────────────────────────────────────────────────
def train_nn(model, loader, device, lr=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(EPOCHS):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx).squeeze()
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
    return model

# ──────────────────────────────────────────────────────────────────────
# MAIN COMPARISON
# ──────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("  COMPARACIÓN MENSUAL FINAL: SARIMAX vs LSTM vs MLP")
    print("="*70)
    
    device = get_device()
    train_df, test_df, s_train, s_test, scaler, feat_list = prepare_monthly_data()
    n_feat = len(feat_list)
    
    # --- 1. SARIMAX ---
    print("\n[1/3] Entrenando SARIMAX...")
    # Usamos seasonal_order de 12 para meses
    sarima_model = SARIMAX(train_df['kWh_Total'], 
                           order=(1,1,1), 
                           seasonal_order=(1,1,1,12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_preds = sarima_fit.get_forecast(steps=6).predicted_mean.values
    
    # --- 2. Preparar Secuencias para Redes ---
    X_train, y_train = create_sequences(s_train, INPUT_WINDOW)
    loader = DataLoader(MonthlyDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # --- 3. LSTM ---
    print("[2/3] Entrenando LSTM Mensual...")
    # Usamos una versión pequeña del LSTM (1 capa, 32 unidades)
    lstm_model = SolarLSTM(n_features=n_feat, hidden_size=32, num_layers=1, output_window=1)
    lstm_model = train_nn(lstm_model, loader, device)
    
    # --- 4. MLP ---
    print("[3/3] Entrenando MLP Mensual...")
    mlp_model = SolarMLP(input_size=INPUT_WINDOW * n_feat, hidden_size=64)
    mlp_model = train_nn(mlp_model, loader, device)
    
    # --- 5. Predicción Recursiva para Redes ---
    def predict_recursive(model, initial_context, steps):
        model.eval()
        current_context = initial_context.copy()
        preds = []
        with torch.no_grad():
            for i in range(steps):
                inp = torch.FloatTensor(current_context[-INPUT_WINDOW:]).unsqueeze(0).to(device)
                p = model(inp).item()
                preds.append(p)
                # Para la siguiente, necesitamos las exógenas del futuro (usamos las de s_test)
                # Buscamos el índice en s_test que corresponde al paso i del test
                # El test empieza justo después de s_train
                future_idx = len(s_train) + i
                if future_idx < len(s_test):
                    next_row = s_test[future_idx].copy()
                    next_row[-1] = p # Reemplazamos kWh real por predicción
                    current_context = np.vstack([current_context, next_row])
        return np.array(preds)

    # El contexto inicial para el test es el final del training
    initial_context = s_train[-INPUT_WINDOW:]
    
    lstm_scaled_preds = predict_recursive(lstm_model, s_train, 6)
    mlp_scaled_preds = predict_recursive(mlp_model, s_train, 6)
    
    # Des-escalar
    def denormalize(p_scaled):
        dummy = np.zeros((len(p_scaled), n_feat))
        dummy[:, -1] = p_scaled
        return scaler.inverse_transform(dummy)[:, -1]
    
    lstm_final = denormalize(lstm_scaled_preds)
    mlp_final = denormalize(mlp_scaled_preds)
    
    # --- 6. Resultados ---
    results_df = pd.DataFrame({
        'Mes': test_df['Mes'],
        'Real': test_df['kWh_Total'],
        'SARIMAX': sarima_preds,
        'LSTM': lstm_final,
        'MLP': mlp_final
    })
    
    print("\n" + "-"*70)
    print("  RESULTADOS PREDICCIÓN SEMESTRE FINAL (2023)")
    print("-"*70)
    print(results_df.to_string(index=False))
    
    # Gráfica
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Mes'], results_df['Real'], marker='o', label='Real', color='black', linewidth=2)
    plt.plot(results_df['Mes'], results_df['SARIMAX'], marker='s', label='SARIMAX', linestyle='--')
    plt.plot(results_df['Mes'], results_df['LSTM'], marker='^', label='LSTM', linestyle='--')
    plt.plot(results_df['Mes'], results_df['MLP'], marker='x', label='MLP', linestyle='--')
    
    plt.title('Comparación Mensual Final (Jul-Dic 2023)', fontsize=14, fontweight='bold')
    plt.ylabel('kWh Total')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, 'monthly_comparison_final.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"\n  Gráfica guardada en: {plot_path}")
    print("="*70)

if __name__ == '__main__':
    main()
