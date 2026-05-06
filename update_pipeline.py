import json
import os

notebook_path = 'd:\\Software\\Energy-forecasting\\Pipeline_Execution.ipynb'

# Load original notebook (which I just modified, so I'll reload it to be sure I have the latest content, 
# but actually I want to revert it to a clean state if possible or just be careful)
# Since I just overwrote it, I'll use the cells that I know are there.
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Re-defining components
setup_code = """import os
import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

# Global Configuration
warnings.filterwarnings('ignore')
sns.set_theme(style=\"whitegrid\")

PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
EDA_DIR = os.path.join(PROJECT_ROOT, 'EDA')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# Utility functions
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_device():
    return torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")

def add_cyclic_features(df):
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df['hour_sin'] = np.sin(2 * np.pi * df['Fecha_Hora'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Fecha_Hora'].dt.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['Fecha_Hora'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Fecha_Hora'].dt.month / 12)
    return df

def compute_metrics_strat(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    m = mape(y_true, y_pred)
    return mae, rmse, m
"""

architecture_code = """class SolarMLP(nn.Module):
    def __init__(self, input_size, hidden_size=256, dropout=0.3):
        super(SolarMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        return self.output(x)

class SolarLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2, output_window=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_window)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :])
        return self.fc(out)

def train_xgboost(X_train, y_train):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train_flat, y_train)
    return model

def predict_xgboost_recursive(model, initial_context, steps, s_test, s_train, input_window, n_feat):
    current_context = initial_context.copy()
    preds = []
    for i in range(steps):
        inp = current_context[-input_window:].reshape(1, -1)
        p = model.predict(inp)[0]
        preds.append(p)
        future_idx = len(s_train) + i
        if future_idx < len(s_test):
            next_row = s_test[future_idx].copy()
            next_row[-1] = p
            current_context = np.vstack([current_context, next_row])
    return np.array(preds)
"""

xm_context_md = """# Contexto de los Datos: XM (Generación Real)

Basándonos en la documentación del proyecto, los datos de generación provienen de **XM**, el operador del Sistema Interconectado Nacional (SIN) en Colombia. Estos registros representan la \"verdad de campo\" (ground truth) obtenida mediante sensores de medición directa instalados en la planta solar **\"El Paso\"**.

### Detalles del Dataset:
- **Fuente**: XM (Telemedida directa).
- **Rango Temporal**: 2019-01-01 a 2023-12-31.
- **Resolución**: Horaria (24 columnas de generación por cada día).
- **Unidad**: kWh.

A continuación, realizaremos la descarga y el análisis exploratorio inicial de estos datos."""

step1_code = """# Descarga de datos desde Google Drive
file_id = '1XM7xrH4WCh1knxF7kTt9Mj3mx1RUGnWC'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
csv_path = os.path.join(DATA_DIR, 'data_2019_2023_kwH_xm.csv')

if not os.path.exists(csv_path):
    print(f\"Descargando datos de XM desde Google Drive...\")
    try:
        response = requests.get(url)
        with open(csv_path, 'wb') as f:
            f.write(response.content)
        print(\"Descarga completada.\")
    except Exception as e:
        print(f\"Error en la descarga: {e}\")
else:
    print(\"El archivo de datos de XM ya existe localmente.\")

# Carga y Procesamiento
df = pd.read_csv(csv_path)
df['Fecha'] = pd.to_datetime(df['Fecha'], format='mixed')
hours = [str(i) for i in range(24)]

# Transformación a formato largo (melted) para análisis
df_melted = df.melt(id_vars=['Fecha'], value_vars=hours, var_name='Hora', value_name='Generacion')
df_melted['Hora'] = df_melted['Hora'].astype(int)
df_melted['Timestamp'] = df_melted['Fecha'] + pd.to_timedelta(df_melted['Hora'], unit='h')
df_melted = df_melted.sort_values('Timestamp').reset_index(drop=True)

# Generación de Gráficas Solicitadas
output_xm_dir = os.path.join(EDA_DIR, 'XM')
os.makedirs(output_xm_dir, exist_ok=True)

print(\"Generando visualizaciones...\")

# 1. Serie temporal completa (Horaria)
plt.figure(figsize=(15, 6))
plt.plot(df_melted['Timestamp'], df_melted['Generacion'], color='#1f77b4', linewidth=0.2)
plt.title('Serie Temporal Completa: Generación Horaria (2019-2023)')
plt.xlabel('Fecha')
plt.ylabel('Generación (kWh)')
plt.savefig(os.path.join(output_xm_dir, 'serie_temporal_completa.png'))
plt.show()

# 2. Serie temporal mensual (Suma total)
df_monthly = df_melted.set_index('Timestamp')['Generacion'].resample('ME').sum().reset_index()
plt.figure(figsize=(15, 6))
plt.plot(df_monthly['Timestamp'], df_monthly['Generacion'], marker='o', color='#e377c2')
plt.title('Serie Temporal: Generación Total Mensual (kWh)')
plt.savefig(os.path.join(output_xm_dir, 'serie_temporal_mensual_total.png'))
plt.show()

# 3. Heatmap día vs hora
heatmap_data = df.set_index('Fecha')[hours].astype(float)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'kWh'})
plt.title('Heatmap de Generación: Día vs Hora')
plt.savefig(os.path.join(output_xm_dir, 'heatmap_dia_vs_hora.png'))
plt.show()

# 4. Boxplot por hora
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_melted, x='Hora', y='Generacion', color='lightblue')
plt.title('Distribución de Generación por Hora (Boxplot)')
plt.savefig(os.path.join(output_xm_dir, 'boxplot_generacion_hora.png'))
plt.show()

# 5. Promedio de generación por hora (Perfil Diario)
promedio_hora = df_melted.groupby('Hora')['Generacion'].mean()
plt.figure(figsize=(10, 5))
sns.lineplot(x=promedio_hora.index, y=promedio_hora.values, marker=\"o\", color=\"#1f77b4\")
plt.title('Promedio de Generación por Hora (Perfil Diario)')
plt.xticks(range(24))
plt.savefig(os.path.join(output_xm_dir, 'promedio_generacion_hora.png'))
plt.show()
"""

step1_exp_md = """### Análisis de las Visualizaciones
1. **Serie Temporal Completa**: Muestra la evolución horaria de la generación a lo largo de 5 años, permitiendo observar la consistencia del recurso.
2. **Serie Temporal Mensual**: Facilita la observación de la estacionalidad anual en la planta solar.
3. **Heatmap Día vs Hora**: Permite identificar patrones de generación a lo largo del día y posibles fallos o mantenimientos en fechas específicas.
4. **Boxplot por Hora**: Muestra la variabilidad de la generación en cada hora específica, resaltando la incertidumbre climática.
5. **Promedio de Generación por Hora**: Define el perfil típico de generación de la planta, confirmando la campana de producción solar.

### Metodología: Ventana Diurna
Como se observa en el perfil diario, la generación significativa ocurre entre las **06:00 AM y las 05:00 PM**. Por ello, el pipeline utiliza este filtro de 12 horas diarias para optimizar el entrenamiento de los modelos, eliminando los periodos nocturnos de generación nula que podrían sesgar las métricas."""

# Define steps by identifying them in the original cells
def get_original_cell(marker):
    for cell in nb['cells']:
        source = "".join(cell['source'])
        if marker in source:
            return cell
    return None

# Step 2 Marker: NASA-XM correlation
step2_cell = get_original_cell("Iniciando Análisis de Correlación NASA-XM")
# Step 3 Marker: SARIMAX strategies
step3_cell = get_original_cell("Iniciando Estrategias de Validación SARIMAX")
# Step 4 Marker: Final Monthly Comparison
step4_cell = get_original_cell("Iniciando Comparación Mensual Final")
# Step 5 Marker: Final Daily Comparison
step5_cell = get_original_cell("Iniciando Comparación Diaria Final")

# New notebook structure
final_cells = []
final_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["# Pipeline de Ejecución: Pronóstico de Energía Solar\\n\\nEste notebook consolida el flujo completo de investigación, desde el análisis de datos XM hasta la comparativa final de modelos."]})
final_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [setup_code]})
final_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Arquitecturas de Modelos Deep Learning y ML"]})
final_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [architecture_code]})
final_cells.append({"cell_type": "markdown", "metadata": {}, "source": [xm_context_md]})
final_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [step1_code]})
final_cells.append({"cell_type": "markdown", "metadata": {}, "source": [step1_exp_md]})

# Add Step 2
if step2_cell:
    final_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Paso 2: Análisis de Correlación NASA-XM"]})
    source = step2_cell['source']
    cleaned_source = [l for l in source if not l.strip().startswith(('import ', 'from ', 'PROJECT_ROOT =', 'DATA_DIR =', 'EDA_DIR =', 'os.makedirs', 'warnings.', 'sns.set_theme'))]
    final_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cleaned_source})

# Add Step 3
if step3_cell:
    final_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Paso 3: Estrategias de Validación SARIMAX"]})
    source = step3_cell['source']
    cleaned_source = [l for l in source if not l.strip().startswith(('import ', 'from ', 'PROJECT_ROOT =', 'DATA_DIR =', 'EDA_DIR =', 'os.makedirs', 'warnings.', 'sns.set_theme'))]
    final_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cleaned_source})

# Add Step 4
if step4_cell:
    final_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Paso 4: Comparación Mensual Final"]})
    source = step4_cell['source']
    cleaned_source = [l for l in source if not l.strip().startswith(('import ', 'from ', 'PROJECT_ROOT =', 'DATA_DIR =', 'EDA_DIR =', 'os.makedirs', 'warnings.', 'sns.set_theme'))]
    final_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cleaned_source})

# Add Step 5
if step5_cell:
    final_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Paso 5: Comparación Diaria Final"]})
    source = step5_cell['source']
    cleaned_source = [l for l in source if not l.strip().startswith(('import ', 'from ', 'PROJECT_ROOT =', 'DATA_DIR =', 'EDA_DIR =', 'os.makedirs', 'warnings.', 'sns.set_theme'))]
    final_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cleaned_source})

# Save
nb['cells'] = final_cells
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook reconstruido correctamente.")
