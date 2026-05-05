import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Datos reales y predicciones (obtenidos de la ejecución anterior)
data = {
    'Real': [13325642.26, 13295989.77, 12609645.10, 12131952.98, 12756108.73, 14739962.19],
    'SARIMAX': [15161270, 13824740, 13548920, 13713060, 13876350, 15381190],
    'LSTM': [10089660, 9039626, 7870229, 7329999, 6859471, 6898002],
    'MLP': [11012040, 10231850, 10167120, 10439930, 9050432, 9189641]
}

def mape(y_true, y_pred):
    return np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

metrics = []
for model in ['SARIMAX', 'LSTM', 'MLP']:
    y_true = data['Real']
    y_pred = data[model]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    
    metrics.append({
        'Modelo': model,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape_val
    })

df_metrics = pd.DataFrame(metrics)
output_path = r'd:\Software\Energy-forecasting\EDA\MONTHLY_COMPARISON\monthly_metrics.csv'
df_metrics.to_csv(output_path, index=False)

print("Métricas calculadas:")
print(df_metrics.to_string(index=False))
