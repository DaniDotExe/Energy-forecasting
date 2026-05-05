import os
import pandas as pd

def main():
    # Rutas de los archivos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nasa_path = os.path.join(base_dir, '..', 'data', 'nasapower-horario.csv')
    xm_path = os.path.join(base_dir, '..', 'data', 'data_2019_2023_kwH_xm.csv')
    output_path = os.path.join(base_dir, '..', 'data', 'data-hourly.csv')

    print("Cargando datasets...")
    # 1. Cargar datos de NASA POWER
    df_nasa = pd.read_csv(nasa_path)
    df_nasa['Fecha_Hora'] = pd.to_datetime(df_nasa['Fecha_Hora'])

    # 2. Cargar datos de XM
    df_xm = pd.read_csv(xm_path)
    df_xm['Fecha'] = pd.to_datetime(df_xm['Fecha'], format='mixed')

    # 3. Procesar XM: De formato ancho (columnas 0-23) a formato largo (melt)
    # Solo tomamos las horas de 6am a 5pm (columnas '6' a '17')
    hours_to_keep = [str(i) for i in range(6, 18)]
    df_xm_melted = df_xm.melt(
        id_vars=['Fecha'], 
        value_vars=hours_to_keep, 
        var_name='Hora', 
        value_name='kWh'
    )
    
    # Crear el timestamp para cruzar con NASA
    df_xm_melted['Hora'] = df_xm_melted['Hora'].astype(int)
    df_xm_melted['Fecha_Hora'] = df_xm_melted['Fecha'] + pd.to_timedelta(df_xm_melted['Hora'], unit='h')
    
    # 4. Seleccionar solo las columnas necesarias de XM para el merge
    df_xm_final = df_xm_melted[['Fecha_Hora', 'kWh']]

    print(f"Combinando datos (Nasa: {len(df_nasa)} filas, XM procesado: {len(df_xm_final)} filas)...")

    # 5. Cruzar con NASA
    # Usamos inner join para quedarnos solo con las horas que existen en ambos (6am-5pm)
    df_combined = pd.merge(df_nasa, df_xm_final, on='Fecha_Hora', how='inner')

    # 6. Guardar el resultado
    print(f"Guardando dataset combinado en: {os.path.normpath(output_path)}")
    df_combined.to_csv(output_path, index=False)
    print("¡Listo!")

if __name__ == "__main__":
    main()
