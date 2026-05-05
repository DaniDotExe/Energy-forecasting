import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def check_stationarity(timeseries):
    """
    Realiza la prueba de Augmented Dickey-Fuller (ADF) para verificar la estacionariedad.
    """
    print("--------------------------------------------------")
    print("Resultados de la Prueba de Dickey-Fuller Aumentada (ADF):")
    # Realizamos el test
    adf_test = adfuller(timeseries, autolag='AIC')
    
    # Formateamos los resultados
    adf_output = pd.Series(adf_test[0:4], index=['Estadístico de prueba (ADF)', 'p-value', '# Lags usados', 'Número de observaciones usadas'])
    for key, value in adfuller(timeseries)[4].items():
        adf_output[f'Valor Crítico ({key})'] = value
        
    print(adf_output)
    print("--------------------------------------------------")
    if adf_test[1] <= 0.05:
        print("Conclusión: Fuerte evidencia contra la hipótesis nula (p <= 0.05).")
        print("La serie de tiempo ES ESTACIONARIA.")
    else:
        print("Conclusión: Débil evidencia contra la hipótesis nula (p > 0.05).")
        print("La serie de tiempo NO ES ESTACIONARIA.")
    print("--------------------------------------------------")

def main():
    # Rutas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'data-hourly.csv')
    output_dir = os.path.join(base_dir, '..', 'EDA', 'SARIMAX')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Cargando datos...")
    df = pd.read_csv(data_path)
    
    # Convertir Fecha_Hora y ordenar
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df = df.sort_values('Fecha_Hora')
    
    # Trabajaremos con la variable objetivo: kWh
    # Como tenemos datos de 06:00 a 17:00 (12 registros por día), 
    # la estacionalidad diaria (periodo) es de 12.
    periodo_estacional = 12
    
    ts = df['kWh'].values
    
    # 1. Prueba de Estacionariedad (ADF)
    check_stationarity(ts)
    
    # Configuración general de los gráficos
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 2. Descomposición Estacional
    # Usamos freq=12 porque nuestros "días" en este dataset filtrado tienen 12 horas.
    print(f"Calculando descomposición estacional (periodo={periodo_estacional})...")
    decomposition = seasonal_decompose(ts, model='additive', period=periodo_estacional)
    
    fig_dec = decomposition.plot()
    fig_dec.set_size_inches(12, 8)
    fig_dec.suptitle('Descomposición Estacional de la Generación (kWh)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'descomposicion_estacional.png'), dpi=300)
    plt.close()
    
    # 3. Gráficos de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)
    print("Generando gráficos ACF y PACF...")
    # Calculamos para 48 lags (equivalente a 4 "días" de 12 horas en nuestro dataset)
    lags = 48
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    plot_acf(ts, lags=lags, ax=ax1, title='Autocorrelación (ACF) - Identifica MA (q, Q)')
    ax1.set_xlabel('Rezagos (Lags)')
    ax1.set_ylabel('Correlación')
    
    plot_pacf(ts, lags=lags, ax=ax2, title='Autocorrelación Parcial (PACF) - Identifica AR (p, P)')
    ax2.set_xlabel('Rezagos (Lags)')
    ax2.set_ylabel('Correlación Parcial')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'acf_pacf.png'), dpi=300)
    plt.close()

    print(f"\nAnálisis completo. Gráficas guardadas en: {os.path.normpath(output_dir)}")
    print("\nGuía rápida para SARIMAX (p,d,q)(P,D,Q,s):")
    print("- 'd': Nivel de diferenciación necesario para que la serie sea estacionaria (ver resultado de ADF).")
    print("- 'p': Lags significativos en el PACF antes de que caigan a cero abruptamente.")
    print("- 'q': Lags significativos en el ACF antes de que caigan a cero abruptamente.")
    print(f"- 's': Estacionalidad, que en este dataset filtrado es {periodo_estacional} (12 horas operativas por día).")
    print("- 'P' y 'Q': Patrones similares en los lags múltiples de 's' (ej. 12, 24, 36) en PACF y ACF respectivamente.")

if __name__ == "__main__":
    main()
