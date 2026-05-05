import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Ruta del archivo (relativa a este script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '..', 'data', 'data_2019_2023_kwH_xm.csv')
    
    # Directorio de salida
    output_dir = os.path.join(base_dir, '..', 'EDA', 'XM')
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar los datos
    print(f"Cargando datos desde: {os.path.normpath(file_path)}")
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo en {file_path}")
        return
    
    df = pd.read_csv(file_path)
    
    # Asegurar que Fecha sea datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='mixed')
    
    # Lista de columnas que corresponden a las horas
    hours = [str(i) for i in range(24)]
    
    # Configuración general de estilo
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # Preparación de datos (Melt)
    # ---------------------------------------------------------
    df_melted = df.melt(id_vars=['Fecha'], value_vars=hours, var_name='Hora', value_name='Generacion')
    df_melted['Hora'] = df_melted['Hora'].astype(int)
    df_melted['Timestamp'] = df_melted['Fecha'] + pd.to_timedelta(df_melted['Hora'], unit='h')
    df_melted = df_melted.sort_values('Timestamp').reset_index(drop=True)
    df_melted['Mes'] = df_melted['Timestamp'].dt.month
    
    # ---------------------------------------------------------
    # 1. Serie temporal mensual (Suma total por mes)
    # ---------------------------------------------------------
    df_monthly_sum = df_melted.set_index('Timestamp')['Generacion'].resample('ME').sum().reset_index()
    
    plt.figure(figsize=(15, 6))
    plt.plot(df_monthly_sum['Timestamp'], df_monthly_sum['Generacion'], marker='o', color="#e377c2", linewidth=2)
    plt.title('Serie Temporal: Generación Total Mensual (kWh)', fontsize=14)
    plt.xlabel('Año', fontsize=12)
    plt.ylabel('Generación Total (kWh)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'serie_temporal_mensual_total.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Promedio de Generación DIARIA por mes
    # ---------------------------------------------------------
    df_daily_total = df_melted.groupby('Fecha')['Generacion'].sum().reset_index()
    df_daily_total['Mes'] = df_daily_total['Fecha'].dt.month
    
    promedio_diario_mes = df_daily_total.groupby('Mes')['Generacion'].mean().reset_index()
    meses_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    plt.figure(figsize=(10, 5))
    # Usamos un color fijo (azul suave) para todas las barras
    sns.barplot(data=promedio_diario_mes, x='Mes', y='Generacion', color="skyblue")
    plt.title('Promedio de Generación DIARIA por Mes', fontsize=14)
    plt.xlabel('Mes', fontsize=12)
    plt.ylabel('Generación Diaria Promedio (kWh)', fontsize=12)
    plt.xticks(ticks=range(12), labels=meses_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'promedio_diario_por_mes.png'))
    plt.close()

    # ---------------------------------------------------------
    # 3. Promedio por hora (Perfil Diario)
    # ---------------------------------------------------------
    promedio_hora = df_melted.groupby('Hora')['Generacion'].mean()
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=promedio_hora.index, y=promedio_hora.values, marker="o", color="#1f77b4", linewidth=2)
    plt.title('Promedio de Generación por Hora (Perfil Diario)', fontsize=14)
    plt.xlabel('Hora del Día', fontsize=12)
    plt.ylabel('Generación Promedio (kWh)', fontsize=12)
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'promedio_generacion_hora.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # 4. Heatmap día-hora
    # ---------------------------------------------------------
    heatmap_data = df.copy()
    heatmap_data['Fecha_str'] = heatmap_data['Fecha'].dt.strftime('%Y-%m-%d')
    heatmap_data = heatmap_data.set_index('Fecha_str')[hours].astype(float)
    
    plt.figure(figsize=(12, 12))
    sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Generación (kWh)'})
    plt.title('Heatmap de Generación: Día vs Hora', fontsize=14)
    plt.xlabel('Hora del Día', fontsize=12)
    plt.ylabel('Fecha', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_dia_vs_hora.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # 5. Boxplot por hora
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Hora', y='Generacion', color="lightblue")
    plt.title('Boxplot de Generación por Hora', fontsize=14)
    plt.xlabel('Hora del Día', fontsize=12)
    plt.ylabel('Generación (kWh)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_generacion_hora.png'))
    plt.close()
    
    print(f"Análisis completado. Gráficas actualizadas en: {os.path.normpath(output_dir)}")

if __name__ == "__main__":
    main()
