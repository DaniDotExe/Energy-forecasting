import os
import pandas as pd

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'data-hourly.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'monthly-data.csv')

def generate_monthly_dataset():
    print("-" * 50)
    print("  GENERANDO DATASET MENSUAL")
    print("-" * 50)
    
    # 1. Cargar datos horarios
    df = pd.read_csv(INPUT_PATH)
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    
    # 2. Crear columna de Mes para agrupar (YYYY-MM)
    df['Mes'] = df['Fecha_Hora'].dt.to_period('M')
    
    # 3. Definir agregaciones
    # Para el kWh queremos la SUMA
    # Para el resto queremos MEAN y STD
    agg_map = {
        'Irradiancia_It': ['mean', 'std'],
        'Temperatura_Tt': ['mean', 'std'],
        'Humedad_Ht': ['mean', 'std'],
        'Viento_Wt': ['mean', 'std'],
        'kWh': 'sum'
    }
    
    # 4. Agrupar
    df_monthly = df.groupby('Mes').agg(agg_map)
    
    # 5. Aplanar los nombres de las columnas (MultiIndex a SingleIndex)
    # Ejemplo: ('Irradiancia_It', 'mean') -> 'Irradiancia_It_mean'
    df_monthly.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] != '' else col[0]
        for col in df_monthly.columns.values
    ]
    
    # Renombrar la columna sum de kWh simplemente a kWh_Total
    df_monthly = df_monthly.rename(columns={'kWh_sum': 'kWh_Total'})
    
    # 6. Resetear índice para que 'Mes' sea una columna
    df_monthly = df_monthly.reset_index()
    df_monthly['Mes'] = df_monthly['Mes'].astype(str)
    
    # 7. Asegurar que kWh_Total sea la última columna (según pedido)
    cols = [c for c in df_monthly.columns if c != 'kWh_Total'] + ['kWh_Total']
    df_monthly = df_monthly[cols]
    
    # 8. Guardar
    df_monthly.to_csv(OUTPUT_PATH, index=False)
    
    print(f"  Proceso completado.")
    print(f"  Archivo guardado en: {os.path.abspath(OUTPUT_PATH)}")
    print(f"  Total de meses procesados: {len(df_monthly)}")
    print("-" * 50)
    print(df_monthly.head())

if __name__ == '__main__':
    generate_monthly_dataset()
