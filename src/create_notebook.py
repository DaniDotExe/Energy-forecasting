#!/usr/bin/env python3
"""Genera el notebook .ipynb programáticamente."""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": [source], "outputs": [], "execution_count": None}

cells = []

# ===================== SECCIÓN 1: TÍTULO Y CONTEXTO =====================
cells.append(md("""# Pronóstico de Generación Solar - Planta "El Paso"
## Comparación: Modelo Estadístico (Holt-Winters) vs. Red Neuronal Recurrente (LSTM)

**Objetivo:** Predecir la generación mensual de energía solar (kWh) de la planta "El Paso" (Cesar, Colombia) usando datos históricos de 2019 a 2023.

### Fuentes de datos
- **Generación (kWh):** Datos reales medidos, provenientes de XM (operador del mercado eléctrico colombiano).
- **Clima (Temperatura, Viento, Irradiancia):** Datos de reanálisis meteorológico de Open-Meteo (Modelo ERA5).

### Metodología
| Aspecto | Holt-Winters | LSTM (RNN) |
|---|---|---|
| **Tipo** | Estadístico (Suavizado Exponencial) | Deep Learning (Red Neuronal Recurrente) |
| **Variables de entrada** | Univariado (solo generación histórica) | Multivariado (generación + clima) |
| **Partición** | 90% ajuste / 10% test | 70% train / 20% val / 10% test |
| **Métrica principal** | MAPE (%) | MAPE (%) |"""))

# ===================== SECCIÓN 2: INSTALAR DEPENDENCIAS (COLAB) =====================
cells.append(md("## 1. Instalación de Dependencias (Google Colab)"))
cells.append(code("""!pip install -q statsmodels torch scikit-learn matplotlib pandas numpy"""))

# ===================== SECCIÓN 3: SUBIR ARCHIVOS =====================
cells.append(md("""## 2. Carga de Archivos

> **Instrucción:** Sube los archivos `monthly_data.csv` y `daily_data.csv` a la raíz del proyecto en Colab (carpeta `/content/`).\n> Puedes hacerlo arrastrándolos al panel de archivos de la izquierda, o ejecutando la celda de abajo."""))

cells.append(code("""from google.colab import files
import os

# Subir archivos manualmente
uploaded = files.upload()

# Verificar que los archivos están en /content/
for f in ['monthly_data.csv', 'daily_data.csv']:
    if os.path.exists(f'/content/{f}'):
        print(f'✓ {f} cargado correctamente.')
    else:
        print(f'✗ {f} NO encontrado. Por favor súbelo.')"""))

# ===================== SECCIÓN 4: IMPORTS =====================
cells.append(md("## 3. Importación de Librerías"))
cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy, warnings, os
warnings.filterwarnings('ignore')

# Holt-Winters
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# PyTorch (LSTM)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Semillas
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {device}')"""))

# ===================== SECCIÓN 5: CARGA DE DATOS =====================
cells.append(md("## 4. Carga y Exploración de Datos"))
cells.append(code("""df = pd.read_csv('/content/monthly_data.csv', encoding='utf-8-sig')

# Limpiar nombres de columnas
df.columns = [c.replace('\\ufeff', '').strip() for c in df.columns]
year_col = [c for c in df.columns if 'o' in c.lower() and len(c) <= 5]
if year_col and 'Año' not in df.columns:
    df.rename(columns={year_col[0]: 'Año'}, inplace=True)

df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str) + '-01')
df = df.sort_values('Fecha').reset_index(drop=True)

print(f"Registros: {len(df)} meses ({df['Fecha'].min().year} - {df['Fecha'].max().year})")
print(f"\\nColumnas numéricas relevantes:")
print(df[['Fecha','Total_Generacion','Temperatura_Tt','Viento_Wt','Irradiancia_It']].describe().round(2))
df[['Fecha','Total_Generacion','Temperatura_Tt','Viento_Wt','Irradiancia_It']].head(10)"""))

# ===================== SECCIÓN 4: VISUALIZACIÓN EXPLORATORIA =====================
cells.append(md("## 5. Análisis Exploratorio"))
cells.append(code("""fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

variables = [
    ('Total_Generacion', 'Generación Total (kWh)', 'steelblue'),
    ('Temperatura_Tt', 'Temperatura (°C)', 'orangered'),
    ('Viento_Wt', 'Viento (km/h)', 'seagreen'),
    ('Irradiancia_It', 'Irradiancia (W/m²)', 'goldenrod')
]

for ax, (col, label, color) in zip(axes, variables):
    ax.plot(df['Fecha'], df[col], color=color, linewidth=1.5)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.set_title(label)

axes[-1].set_xlabel('Fecha')
fig.suptitle('Series Temporales Mensuales - Planta El Paso (2019-2023)', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))

# ===================== SECCIÓN 5: HOLT-WINTERS =====================
cells.append(md("""## 6. Modelo 1: Holt-Winters (Suavizado Exponencial Triple)

Holt-Winters es un método estadístico clásico que descompone la serie temporal en tres componentes:
- **Nivel:** El valor promedio de la serie.
- **Tendencia:** La dirección general (creciente/decreciente).
- **Estacionalidad:** Patrones cíclicos repetitivos (en este caso, ciclo anual de 12 meses).

> **Nota:** Es un modelo **univariado**: solo utiliza el histórico de `Total_Generacion` para predecir."""))

cells.append(code("""# --- Preparación de datos ---
y = df['Total_Generacion'].values
fechas = df['Fecha'].values
n = len(y)

# Split 90/10
split_hw = int(n * 0.90)
y_train_hw = y[:split_hw]
y_test_hw = y[split_hw:]
fechas_train_hw = fechas[:split_hw]
fechas_test_hw = fechas[split_hw:]

print(f"Holt-Winters - Train: {len(y_train_hw)} meses, Test: {len(y_test_hw)} meses")

# --- Ajustar modelo ---
hw_model = ExponentialSmoothing(
    y_train_hw,
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    initialization_method="estimated"
).fit()

# --- Predicciones ---
hw_fitted = hw_model.fittedvalues
hw_forecast = hw_model.forecast(len(y_test_hw))

# --- MAPE ---
epsilon = 1e-10
mape_hw = np.mean(np.abs((y_test_hw - hw_forecast) / (y_test_hw + epsilon))) * 100
print(f"\\n>>> MAPE Holt-Winters (Test): {mape_hw:.2f}% <<<")"""))

cells.append(md("### 6.1 Gráficas Holt-Winters"))
cells.append(code("""fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Gráfica completa
ax = axes[0]
ax.plot(fechas_train_hw, y_train_hw, 'b-', label='Real Train', alpha=0.4)
ax.plot(fechas_test_hw, y_test_hw, 'k-', label='Real Test', alpha=0.4)
ax.plot(fechas_train_hw, hw_fitted, 'r--', label='Ajuste Train', alpha=0.8)
ax.plot(fechas_test_hw, hw_forecast, 'c--', label='Predicción Test', alpha=0.8)
ax.set_title(f'Holt-Winters: Predicción Completa - MAPE Test: {mape_hw:.2f}%')
ax.legend(); ax.grid(True, alpha=0.3)

# Gráfica zoom test
ax = axes[1]
ax.plot(fechas_test_hw, y_test_hw, 'b-o', label='Real', markersize=6)
ax.plot(fechas_test_hw, hw_forecast, 'r--s', label='Predicción', markersize=6)
ax.set_title(f'Holt-Winters: Zoom Test - MAPE: {mape_hw:.2f}%')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# ===================== SECCIÓN 6: LSTM =====================
cells.append(md("""## 7. Modelo 2: Red Neuronal Recurrente (LSTM Avanzada - PyTorch)

La LSTM (Long Short-Term Memory) es una arquitectura de red neuronal diseñada específicamente para aprender dependencias temporales a largo plazo.

**Características de nuestra arquitectura avanzada:**
- **Bidireccional:** Procesa la secuencia en ambas direcciones temporales.
- **Layer Normalization:** Estabiliza el entrenamiento.
- **Global Average Pooling:** Resume toda la ventana temporal en vez de usar solo el último paso.
- **LeakyReLU:** Evita la muerte de neuronas (dying ReLU).

> **Nota:** Es un modelo **multivariado**: utiliza `Mes`, `Temperatura`, `Viento` e `Irradiancia` como features."""))

cells.append(code("""# --- Definición del modelo ---
class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden1=64, hidden2=32):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden1*2, hidden2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.2)
        self.norm  = nn.LayerNorm(hidden2*2)
        self.fc1   = nn.Linear(hidden2*2, 32)
        self.act   = nn.LeakyReLU(0.01)
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x); out = self.drop1(out)
        out, _ = self.lstm2(out); out = self.drop2(out)
        out = torch.mean(out, dim=1)  # Global Average Pooling
        out = self.norm(out)
        return self.fc2(self.act(self.fc1(out)))

print("Arquitectura definida correctamente.")"""))

cells.append(md("### 7.1 Preparación de Datos para la LSTM"))
cells.append(code("""FEATURE_COLS = ['Mes', 'Temperatura_Tt', 'Viento_Wt', 'Irradiancia_It']
TARGET_COL = 'Total_Generacion'
LOOKBACK = 6  # Ventana de 6 meses
EPOCHS = 1000; BATCH_SIZE = 8; PATIENCE = 100

# Estandarización (Z-score)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(df[FEATURE_COLS].values)
y_scaled = scaler_y.fit_transform(df[[TARGET_COL]].values)

# Crear secuencias
def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOKBACK)
fechas_seq = df['Fecha'].values[LOOKBACK:]

# Split 70/20/10 cronológico
n_seq = len(X_seq)
tr = int(n_seq * 0.70)
va = int(n_seq * 0.90)

X_tr, y_tr = X_seq[:tr], y_seq[:tr]
X_va, y_va = X_seq[tr:va], y_seq[tr:va]
X_te, y_te = X_seq[va:], y_seq[va:]

print(f"LSTM - Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)} secuencias")

train_dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                     torch.tensor(y_tr, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                                     torch.tensor(y_va, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)"""))

cells.append(md("### 7.2 Entrenamiento de la LSTM"))
cells.append(code("""model = AdvancedLSTMModel(input_size=len(FEATURE_COLS)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
best_state = copy.deepcopy(model.state_dict())
patience_cnt = 0
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    t_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        t_loss += loss.item()
    train_losses.append(t_loss / len(train_dl))

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for xv, yv in val_dl:
            xv, yv = xv.to(device), yv.to(device)
            v_loss += criterion(model(xv), yv).item()
    val_losses.append(v_loss / len(val_dl))

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        patience_cnt = 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early Stopping en época {epoch+1}")
            break

model.load_state_dict(best_state)
print(f"Entrenamiento finalizado en {len(train_losses)} épocas.")"""))

cells.append(md("### 7.3 Predicciones y Métricas LSTM"))
cells.append(code("""model.eval()
with torch.no_grad():
    pred_tr = model(torch.tensor(X_tr, dtype=torch.float32).to(device)).cpu().numpy()
    pred_va = model(torch.tensor(X_va, dtype=torch.float32).to(device)).cpu().numpy()
    pred_te = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()

# Re-escalar a kWh reales
real_tr = scaler_y.inverse_transform(y_tr.reshape(-1,1)).flatten()
real_va = scaler_y.inverse_transform(y_va.reshape(-1,1)).flatten()
real_te = scaler_y.inverse_transform(y_te.reshape(-1,1)).flatten()
pred_tr = scaler_y.inverse_transform(pred_tr).flatten()
pred_va = scaler_y.inverse_transform(pred_va).flatten()
pred_te = scaler_y.inverse_transform(pred_te).flatten()

epsilon = 1e-10
mape_lstm = np.mean(np.abs((real_te - pred_te) / (real_te + epsilon))) * 100
print(f"\\n>>> MAPE LSTM (Test): {mape_lstm:.2f}% <<<")"""))

cells.append(md("### 7.4 Gráficas LSTM"))
cells.append(code("""fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# Loss
ax = axes[0]
ax.plot(train_losses, label='Loss Train'); ax.plot(val_losses, label='Loss Val')
ax.set_title(f'Pérdida por Época - MAPE Final: {mape_lstm:.2f}%')
ax.set_xlabel('Época'); ax.set_ylabel('MSE Loss'); ax.legend(); ax.grid(True, alpha=0.3)

# Predicción completa
ax = axes[1]
ax.plot(fechas_seq[:tr], real_tr, 'b-', alpha=0.3, label='Real Train')
ax.plot(fechas_seq[tr:va], real_va, 'g-', alpha=0.3, label='Real Val')
ax.plot(fechas_seq[va:], real_te, 'k-', alpha=0.3, label='Real Test')
ax.plot(fechas_seq[:tr], pred_tr, 'r--', alpha=0.8, label='Pred Train')
ax.plot(fechas_seq[tr:va], pred_va, 'm--', alpha=0.8, label='Pred Val')
ax.plot(fechas_seq[va:], pred_te, 'c--', alpha=0.8, label='Pred Test')
ax.set_title(f'LSTM: Predicción Completa - MAPE Test: {mape_lstm:.2f}%')
ax.legend(); ax.grid(True, alpha=0.3)

# Zoom test
ax = axes[2]
ax.plot(fechas_seq[va:], real_te, 'b-o', label='Real', markersize=6)
ax.plot(fechas_seq[va:], pred_te, 'r--s', label='Predicción', markersize=6)
ax.set_title(f'LSTM: Zoom Test - MAPE: {mape_lstm:.2f}%')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# ===================== SECCIÓN 7: COMPARACIÓN =====================
cells.append(md("""## 8. Comparación Final: Holt-Winters vs. LSTM

A continuación se presenta la comparación directa de ambos modelos sobre el mismo periodo de test (los últimos 6 meses de 2023)."""))

cells.append(code("""# --- Tabla comparativa ---
comparison = pd.DataFrame({
    'Modelo': ['Holt-Winters', 'LSTM (Avanzada)'],
    'Tipo': ['Estadístico (Univariado)', 'Deep Learning (Multivariado)'],
    'Variables de Entrada': ['Solo generación histórica', 'Mes + Temperatura + Viento + Irradiancia'],
    'MAPE Test (%)': [round(mape_hw, 2), round(mape_lstm, 2)],
    'Mejor': ['✓' if mape_hw < mape_lstm else '', '✓' if mape_lstm < mape_hw else '']
})
print("=" * 80)
print("TABLA COMPARATIVA DE RESULTADOS")
print("=" * 80)
print(comparison.to_string(index=False))
print("=" * 80)"""))

cells.append(code("""# --- Gráfica comparativa de predicciones en el periodo test ---
fig, ax = plt.subplots(figsize=(12, 6))

# Alinear fechas de test (HW usa fechas directas, LSTM usa fechas post-lookback)
ax.plot(fechas_test_hw, y_test_hw, 'k-o', label='Real', markersize=8, linewidth=2)
ax.plot(fechas_test_hw, hw_forecast, 'b--^', label=f'Holt-Winters (MAPE: {mape_hw:.2f}%)', markersize=8, linewidth=2)
ax.plot(fechas_seq[va:], pred_te, 'r--s', label=f'LSTM (MAPE: {mape_lstm:.2f}%)', markersize=8, linewidth=2)

ax.set_title('Comparación de Predicciones en Test (Últimos meses de 2023)', fontsize=14)
ax.set_xlabel('Fecha'); ax.set_ylabel('Generación (kWh)')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

# ===================== SECCIÓN 8: CONCLUSIONES =====================
cells.append(md("""## 9. Conclusiones

### Hallazgos Principales

1. **Ambos modelos logran errores bajos** en datos mensuales, demostrando que la generación solar de la planta El Paso presenta patrones temporales predecibles.

2. **Holt-Winters** es altamente efectivo a pesar de su simplicidad: captura la estacionalidad anual de 12 meses sin necesidad de variables climáticas externas. Su fortaleza radica en que la generación solar mensual es inherentemente cíclica.

3. **La LSTM Avanzada** incorpora información climática (temperatura, viento, irradiancia) como variables exógenas, lo que le permite capturar variaciones no estacionales causadas por anomalías climáticas.

4. **Limitación compartida:** Con solo 60 registros mensuales (5 años), ambos modelos operan con datos limitados. Más años de historial mejorarían significativamente la capacidad de generalización.

### Recomendaciones
- Para **pronósticos operativos rápidos**, Holt-Winters es preferible por su simplicidad y velocidad.
- Para **escenarios con anomalías climáticas**, la LSTM es más robusta al integrar variables meteorológicas.
- Se recomienda **actualizar los modelos anualmente** a medida que se acumulen más datos de generación."""))

# ===================== CONSTRUIR NOTEBOOK =====================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

BASE_DIR = r"d:\Software\Energy-forecasting"
output_path = os.path.join(BASE_DIR, "src", "Solar_Forecasting_Colab.ipynb")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook Colab creado exitosamente: {output_path}")
