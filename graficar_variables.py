#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir = r"d:\Software\Energy-forecasting"
daily_file = os.path.join(base_dir, "daily_data.csv")
monthly_file = os.path.join(base_dir, "monthly_data.csv")

# Directorio para guardar las gráficas
plots_dir = os.path.join(base_dir, "graficas")
os.makedirs(plots_dir, exist_ok=True)

# Configuración de las variables climáticas (nombre para el eje y color)
weather_vars = {
    'Temperatura_Tt': ('Temperatura (°C)', 'red'),
    'Viento_Wt': ('Velocidad del Viento (km/h)', 'green'),
    'Irradiancia_It': ('Irradiancia (W/m²)', 'orange')
}

def plot_data(df, time_col, gen_col, title_prefix, filename_prefix):
    # Asegurarnos de que esté ordenado por tiempo
    df = df.sort_values(by=time_col)
    
    for var, (var_label, color) in weather_vars.items():
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Eje Y principal (Izquierda) para la Generación
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Total Generación (kWh)', color='blue', fontweight='bold')
        # Usamos alpha para que la línea no tape por completo a la otra
        ax1.plot(df[time_col], df[gen_col], color='blue', alpha=0.5, label='Generación')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Eje Y secundario (Derecha) para la variable climática
        ax2 = ax1.twinx()
        ax2.set_ylabel(var_label, color=color, fontweight='bold')
        ax2.plot(df[time_col], df[var], color=color, alpha=0.8, label=var)
        ax2.tick_params(axis='y', labelcolor=color)

        # Título y diseño
        plt.title(f'{title_prefix}: Generación vs {var}')
        fig.tight_layout()
        
        # Guardar imagen
        out_path = os.path.join(plots_dir, f"{filename_prefix}_{var}.png")
        plt.savefig(out_path, dpi=300)
        plt.close() # Cerrar la figura para liberar memoria
        
        print(f"  -> Gráfica generada: {out_path}")

# ==========================================
# 1. Procesar Gráficas Diarias
# ==========================================
print("Generando gráficas para datos diarios...")
try:
    df_daily = pd.read_csv(daily_file)
    df_daily['Fecha'] = pd.to_datetime(df_daily['Fecha'])
    plot_data(df_daily, 'Fecha', 'Total_Generacion', 'Datos Diarios', 'diario')
except Exception as e:
    print(f"Error procesando datos diarios: {e}")

# ==========================================
# 2. Procesar Gráficas Mensuales
# ==========================================
print("\nGenerando gráficas para datos mensuales...")
try:
    df_monthly = pd.read_csv(monthly_file)
    # Crear una columna de fecha sintética (el día 1 de cada mes) para poder graficar como serie de tiempo
    df_monthly['Fecha'] = pd.to_datetime(df_monthly['Año'].astype(str) + '-' + df_monthly['Mes'].astype(str) + '-01')
    plot_data(df_monthly, 'Fecha', 'Total_Generacion', 'Datos Mensuales', 'mensual')
except Exception as e:
    print(f"Error procesando datos mensuales: {e}")

print("\n=== Proceso finalizado. Todas las gráficas están en la carpeta 'graficas'. ===")
