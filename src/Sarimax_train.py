"""
SARIMAX Model Training & Comparison
====================================
Entrena 5 configuraciones SARIMAX sobre data-hourly.csv (kWh):
  1. SARIMAX(1,0,1)(1,0,1)_12
  2. SARIMAX(2,0,2)(1,0,1)_12
  3. SARIMAX(2,0,2)(1,1,1)_12
  4. SARIMAX(2,0,1)(1,1,1)_12
  5. AUTO ARIMA (pmdarima)

Split: 70% train | 20% validación | 10% test
  - Train:      2019-01-01 06:00 -> 2022-07-04 13:00
  - Validación: 2022-07-04 14:00 -> 2023-07-03 15:00
  - Test:       2023-07-03 16:00 -> 2023-12-31 17:00

Métricas: RMSE, MAE, MAPE, AIC, BIC
Gráficas: Real vs Predicción, Residuos, ACF Residuos
Resultado: Se selecciona el MEJOR modelo.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'data-hourly.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'EDA', 'SARIMAX_training')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"  Directorio de salida: {os.path.abspath(OUTPUT_DIR)}")

SEASONAL_PERIOD = 12  # 12 horas operativas por día (06:00–17:00)
EXOG_COLS = ['Irradiancia_It', 'Temperatura_Tt', 'Humedad_Ht', 'Viento_Wt']

# Modelos a entrenar: nombre -> (order, seasonal_order)
MODELS_CONFIG = {
    'SARIMAX(1,0,1)(1,0,1)_12': {'order': (1, 0, 1), 'seasonal_order': (1, 0, 1, SEASONAL_PERIOD)},
    'SARIMAX(2,0,2)(1,0,1)_12': {'order': (2, 0, 2), 'seasonal_order': (1, 0, 1, SEASONAL_PERIOD)},
    'SARIMAX(2,0,2)(1,1,1)_12': {'order': (2, 0, 2), 'seasonal_order': (1, 1, 1, SEASONAL_PERIOD)},
    'SARIMAX(2,0,1)(1,1,1)_12': {'order': (2, 0, 1), 'seasonal_order': (1, 1, 1, SEASONAL_PERIOD)},
}


# ──────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    """Mean Absolute Percentage Error, evitando divisiones por cero."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true, y_pred, aic=None, bic=None):
    """Calcula RMSE, MAE, MAPE y opcionalmente AIC/BIC."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return {
        'RMSE': rmse,
        'MAE': mae_val,
        'MAPE (%)': mape_val,
        'AIC': aic,
        'BIC': bic,
    }


def plot_real_vs_pred(y_true, y_pred, title, filepath):
    """Gráfica de serie real vs predicción."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true.index, y_true.values, label='Real', color='#1f77b4', linewidth=0.8, alpha=0.8)
    ax.plot(y_true.index, y_pred, label='Predicción', color='#ff7f0e', linewidth=0.8, alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Fecha-Hora')
    ax.set_ylabel('kWh')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()


def plot_residuals(residuals, title, filepath):
    """Gráfica de residuos."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Residuos en el tiempo
    axes[0].plot(residuals, color='#2ca02c', linewidth=0.5, alpha=0.7)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=0.8)
    axes[0].set_title(f'{title} – Residuos', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Residuo (kWh)')
    axes[0].grid(True, alpha=0.3)

    # Histograma de residuos
    axes[1].hist(residuals, bins=50, color='#9467bd', edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residuo (kWh)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()


def plot_acf_residuals(residuals, title, filepath, lags=48):
    """ACF de los residuos."""
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(residuals, lags=lags, ax=ax, title=f'{title} – ACF Residuos')
    ax.set_xlabel('Lags')
    ax.set_ylabel('Autocorrelación')
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────
def load_data():
    print("=" * 70)
    print("  CARGA DE DATOS")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH)
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df = df.sort_values('Fecha_Hora').reset_index(drop=True)
    df.set_index('Fecha_Hora', inplace=True)

    print(f"  Registros totales : {len(df):,}")
    print(f"  Rango temporal    : {df.index.min()} -> {df.index.max()}")
    print(f"  Columnas          : {list(df.columns)}")
    print(f"  Nulos en kWh      : {df['kWh'].isna().sum()}")
    print()
    return df


# ──────────────────────────────────────────────────────────────────────
# SPLIT 70 / 20 / 10
# ──────────────────────────────────────────────────────────────────────
def split_data(df):
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)
    # n_test = n - n_train - n_val  # el resto (~10%)

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    print("-" * 70)
    print("  DISTRUBUCIÓN DE DATOS (70/20/10)")
    print("-" * 70)
    print(f"  TRAIN : {train.index.min()} -> {train.index.max()} ({len(train):,} registros)")
    print(f"  VALID : {val.index.min()}   -> {val.index.max()} ({len(val):,} registros)")
    print(f"  TEST  : {test.index.min()}  -> {test.index.max()} ({len(test):,} registros)")
    print()

    print("=" * 70)
    print("  DIVISIÓN DEL DATASET")
    print("=" * 70)
    print(f"  Train : {len(train):>6,}  ({len(train)/n*100:.1f}%)  [{train.index.min()} -> {train.index.max()}]")
    print(f"  Val   : {len(val):>6,}  ({len(val)/n*100:.1f}%)  [{val.index.min()} -> {val.index.max()}]")
    print(f"  Test  : {len(test):>6,}  ({len(test)/n*100:.1f}%)  [{test.index.min()} -> {test.index.max()}]")
    print()
    return train, val, test


# ──────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UN MODELO SARIMAX
# ──────────────────────────────────────────────────────────────────────
def train_sarimax(train_series, val_series, name, order, seasonal_order, exog_train=None, exog_val=None):
    """
    Entrena SARIMAX con train, predice de una sola vez todo el bloque de validación.
    """
    print(f"  Entrenando {name} ...")
    t0 = time.time()

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=500)
    elapsed = time.time() - t0
    print(f"    Ajuste completado en {elapsed:.1f}s")

    # Predicción de una sola vez sobre el periodo de validación
    n_val = len(val_series)
    forecast = result.get_forecast(steps=n_val, exog=exog_val)
    y_pred = forecast.predicted_mean.values

    # Métricas
    metrics = compute_metrics(val_series.values, y_pred, aic=result.aic, bic=result.bic)

    # Residuos = real - predicción
    residuals = val_series.values - y_pred

    print(f"    RMSE   = {metrics['RMSE']:>12,.2f}")
    print(f"    MAE    = {metrics['MAE']:>12,.2f}")
    print(f"    MAPE   = {metrics['MAPE (%)']:>11.2f}%")
    print(f"    AIC    = {metrics['AIC']:>12,.2f}")
    print(f"    BIC    = {metrics['BIC']:>12,.2f}")
    print()

    return {
        'name': name,
        'result': result,
        'y_pred': y_pred,
        'residuals': residuals,
        'metrics': metrics,
        'time': elapsed,
    }


# ──────────────────────────────────────────────────────────────────────
# AUTO ARIMA
# ──────────────────────────────────────────────────────────────────────
def train_auto_arima(train_series, val_series, exog_train=None, exog_val=None):
    """
    Usa pmdarima para buscar automáticamente los mejores parámetros SARIMAX.
    """
    try:
        import pmdarima as pm
    except ImportError:
        print("  [!] pmdarima no está instalado. Instalando...")
        os.system(f'{sys.executable} -m pip install pmdarima -q')
        import pmdarima as pm

    print("  Entrenando AUTO ARIMA (esto puede tardar varios minutos) ...")
    t0 = time.time()

    auto_model = pm.auto_arima(
        train_series,
        X=exog_train,
        seasonal=True,
        m=SEASONAL_PERIOD,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        max_p=2, max_q=2,
        max_P=1, max_Q=1,
        max_d=1, max_D=1,
        trace=True,
        n_fits=30,
        method='nm',
    )
    elapsed = time.time() - t0
    print(f"    Auto ARIMA completado en {elapsed:.1f}s")
    print(f"    Mejor orden encontrado: {auto_model.order} x {auto_model.seasonal_order}")

    # Predicción
    n_val = len(val_series)
    y_pred = auto_model.predict(n_periods=n_val, X=exog_val)

    # Métricas
    metrics = compute_metrics(
        val_series.values, y_pred,
        aic=auto_model.aic(), bic=auto_model.bic()
    )

    residuals = val_series.values - y_pred

    print(f"    RMSE   = {metrics['RMSE']:>12,.2f}")
    print(f"    MAE    = {metrics['MAE']:>12,.2f}")
    print(f"    MAPE   = {metrics['MAPE (%)']:>11.2f}%")
    print(f"    AIC    = {metrics['AIC']:>12,.2f}")
    print(f"    BIC    = {metrics['BIC']:>12,.2f}")
    print()

    return {
        'name': f'AUTO ARIMA {auto_model.order}x{auto_model.seasonal_order}',
        'result': auto_model,
        'y_pred': y_pred,
        'residuals': residuals,
        'metrics': metrics,
        'time': elapsed,
    }


# ──────────────────────────────────────────────────────────────────────
# TABLA COMPARATIVA
# ──────────────────────────────────────────────────────────────────────
def print_comparison_table(results):
    print("=" * 90)
    print("  TABLA COMPARATIVA DE MODELOS")
    print("=" * 90)
    header = f"  {'Modelo':<38} {'RMSE':>10} {'MAE':>10} {'MAPE%':>8} {'AIC':>14} {'BIC':>14}"
    print(header)
    print("  " + "-" * 86)
    for r in results:
        m = r['metrics']
        print(f"  {r['name']:<38} {m['RMSE']:>10,.2f} {m['MAE']:>10,.2f} {m['MAPE (%)']:>7.2f}% {m['AIC']:>14,.2f} {m['BIC']:>14,.2f}")
    print()


# ──────────────────────────────────────────────────────────────────────
# SELECCIÓN DEL MEJOR MODELO
# ──────────────────────────────────────────────────────────────────────
def select_best(results):
    """Selecciona el mejor modelo por RMSE más bajo."""
    best = min(results, key=lambda x: x['metrics']['RMSE'])
    print("=" * 70)
    print(f"  [BEST] MEJOR MODELO: {best['name']}")
    print("=" * 70)
    m = best['metrics']
    print(f"    RMSE   = {m['RMSE']:>12,.2f}")
    print(f"    MAE    = {m['MAE']:>12,.2f}")
    print(f"    MAPE   = {m['MAPE (%)']:>11.2f}%")
    print(f"    AIC    = {m['AIC']:>12,.2f}")
    print(f"    BIC    = {m['BIC']:>12,.2f}")
    print()
    return best


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    # 1) Cargar datos
    df = load_data()
    ts = df['kWh']

    # 2) Split
    train, val, test = split_data(df)
    train_ts = train['kWh']
    val_ts = val['kWh']

    # Variables exógenas: Irradiancia, Temperatura, Humedad, Viento
    exog_train = train[EXOG_COLS]
    exog_val = val[EXOG_COLS]

    print("  Variables exogenas: ", EXOG_COLS)
    print()

    # 3) Entrenar los 4 modelos SARIMAX manuales
    results = []
    print("=" * 70)
    print("  ENTRENAMIENTO DE MODELOS SARIMAX (con exogenas)")
    print("=" * 70)

    for name, cfg in MODELS_CONFIG.items():
        r = train_sarimax(
            train_series=train_ts,
            val_series=val_ts,
            name=name,
            order=cfg['order'],
            seasonal_order=cfg['seasonal_order'],
            exog_train=exog_train,
            exog_val=exog_val,
        )
        results.append(r)

    # 4) Auto ARIMA (con exogenas)
    print("-" * 70)
    r_auto = train_auto_arima(train_ts, val_ts, exog_train=exog_train, exog_val=exog_val)
    results.append(r_auto)

    # 5) Tabla comparativa
    print_comparison_table(results)

    # 6) Generar gráficas para cada modelo
    print("Generando gráficas...")
    for r in results:
        safe_name = r['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '-')

        # Real vs Predicción
        plot_real_vs_pred(
            val_ts, r['y_pred'],
            title=f"Real vs Predicción – {r['name']}",
            filepath=os.path.join(OUTPUT_DIR, f'{safe_name}_real_vs_pred.png'),
        )

        # Residuos
        plot_residuals(
            r['residuals'],
            title=r['name'],
            filepath=os.path.join(OUTPUT_DIR, f'{safe_name}_residuos.png'),
        )

        # ACF Residuos
        plot_acf_residuals(
            r['residuals'],
            title=r['name'],
            filepath=os.path.join(OUTPUT_DIR, f'{safe_name}_acf_residuos.png'),
        )
    print(f"  Gráficas guardadas en: {os.path.normpath(OUTPUT_DIR)}")
    print()

    # 7) Seleccionar mejor modelo
    best = select_best(results)

    # 8) Guardar resumen en CSV
    summary_rows = []
    for r in results:
        row = {'Modelo': r['name'], 'Tiempo_s': round(r['time'], 1)}
        row.update(r['metrics'])
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'comparacion_modelos.csv'), index=False)
    print(f"  Resumen guardado en: {os.path.normpath(os.path.join(OUTPUT_DIR, 'comparacion_modelos.csv'))}")

    # 9) Gráfica comparativa de barras (RMSE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = [r['name'] for r in results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (metric_key, ylabel) in enumerate([('RMSE', 'RMSE (kWh)'), ('MAE', 'MAE (kWh)'), ('MAPE (%)', 'MAPE (%)')]):
        vals = [r['metrics'][metric_key] for r in results]
        bars = axes[idx].bar(range(len(names)), vals, color=colors[:len(names)], edgecolor='black', alpha=0.85)
        axes[idx].set_xticks(range(len(names)))
        axes[idx].set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        axes[idx].set_ylabel(ylabel, fontsize=11)
        axes[idx].set_title(metric_key, fontsize=13, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Anotar valores en las barras
        for bar, v in zip(bars, vals):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:,.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold'
            )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparacion_metricas.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Gráfica comparativa guardada.")
    print()
    print("=" * 70)
    print("  ENTRENAMIENTO COMPLETO")
    print("=" * 70)


if __name__ == '__main__':
    main()
