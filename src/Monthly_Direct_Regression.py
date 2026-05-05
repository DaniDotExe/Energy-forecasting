import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'monthly-data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'EDA', 'MONTHLY_REGRESSION')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_direct_regression():
    print("="*70)
    print("  EXPERIMENTO: REGRESIÓN DIRECTA (SIN VENTANA TEMPORAL)")
    print("="*70)
    
    # 1. Cargar datos
    df = pd.read_csv(DATA_PATH)
    
    # Características: Solo clima (mean y std)
    # NO incluimos kWh como entrada
    features = [c for c in df.columns if c not in ['Mes', 'kWh_Total']]
    target = 'kWh_Total'
    
    # 2. Split (90% Train, 10% Test) -> Últimos 6 meses para test
    train_df = df.iloc[:-6]
    test_df = df.iloc[-6:]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # 3. Escalamiento (Opcional para XGBoost pero bueno para consistencia)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Entrenar XGBoost (Regresión Pura)
    print(f"  Entrenando XGBoost con {len(features)} variables climáticas...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model_xgb.fit(X_train_scaled, y_train)
    
    # 5. Predicción Directa (No recursiva, solo pasar el clima del mes)
    preds_xgb = model_xgb.predict(X_test_scaled)
    
    # 6. Resultados y Métricas
    results = pd.DataFrame({
        'Mes': test_df['Mes'],
        'Real': y_test.values,
        'XGB_Direct': preds_xgb
    })
    
    rmse = np.sqrt(mean_squared_error(y_test, preds_xgb))
    mape = np.mean(np.abs((y_test.values - preds_xgb) / y_test.values)) * 100
    
    print("\n" + "-"*70)
    print("  RESULTADOS REGRESIÓN DIRECTA (XGBoost)")
    print("-"*70)
    print(results.to_string(index=False))
    print(f"\n  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # 7. Gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(results['Mes'], results['Real'], marker='o', label='Real', color='black')
    plt.plot(results['Mes'], results['XGB_Direct'], marker='d', label='XGBoost Directo', linestyle='--')
    plt.title('Regresión Directa: Clima del Mes -> kWh del Mes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, 'direct_regression_results.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\n  Gráfica guardada en: {plot_path}")
    print("="*70)

if __name__ == '__main__':
    run_direct_regression()
