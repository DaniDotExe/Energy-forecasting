import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Rutas de archivos y directorios
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'data-hourly.csv')
    output_dir = os.path.join(base_dir, '..', 'EDA', 'NASA_xm_correlation')
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Cargando dataset combinado desde: {os.path.normpath(data_path)}")
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo en {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Seleccionar las variables para la correlación
    cols_to_correlate = {
        'kWh': 'Generación (kWh)',
        'Irradiancia_It': 'Irradiancia',
        'Temperatura_Tt': 'Temperatura',
        'Humedad_Ht': 'Humedad',
        'Viento_Wt': 'Viento'
    }
    
    # ---------------------------------------------------------
    # 1. Matriz de Correlación (Heatmap)
    # ---------------------------------------------------------
    df_corr = df[list(cols_to_correlate.keys())].rename(columns=cols_to_correlate)
    corr_matrix = df_corr.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True)
    plt.title('Matriz de Correlación', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_1_heatmap_correlacion.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Scatter Plot: Irradiancia vs Generación
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Irradiancia_It', y='kWh', alpha=0.5, color='orange')
    plt.title('Irradiancia vs Generación de Energía', fontsize=14)
    plt.xlabel('Irradiancia (NASA POWER)', fontsize=12)
    plt.ylabel('Generación (kWh - XM)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_irradiancia_vs_generacion.png'), dpi=300)
    plt.close()
    
    print(f"Análisis completado. Gráficas guardadas en: {os.path.normpath(output_dir)}")

if __name__ == "__main__":
    main()
