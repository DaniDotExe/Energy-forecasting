#!/usr/bin/env python3
import pandas as pd
import os

base_dir = r"d:\Software\Energy-forecasting"
generation_file = os.path.join(base_dir, "data_2019_2023_kwH_xm.csv")
weather_file = os.path.join(base_dir, "clima_diario_el_paso_diurno.csv")
output_file = os.path.join(base_dir, "data.csv")

print("=== Iniciando creación del dataset final ===")

# 1. Cargar datos de generación
print(f"Cargando datos de generación...")
df_gen = pd.read_csv(generation_file)

# Convertir 'Fecha' a datetime (eliminando horas/minutos en caso de existir)
df_gen['Fecha'] = pd.to_datetime(df_gen['Fecha'], format='mixed').dt.normalize()

# Preparar las columnas de las horas requeridas (6 a 17)
horas_cols = [str(i) for i in range(6, 18)]

# Por precaución, si los nombres de columnas tienen decimales como '6.0' en lugar de '6'
for h in horas_cols:
    if h not in df_gen.columns and f"{h}.0" in df_gen.columns:
        df_gen.rename(columns={f"{h}.0": h}, inplace=True)

cols_to_keep = ['Fecha', 'Recurso'] + horas_cols
df_gen = df_gen[cols_to_keep]

# 2. Cargar datos de clima
print(f"Cargando datos de clima...")
df_weather = pd.read_csv(weather_file)
df_weather['Fecha_Diaria'] = pd.to_datetime(df_weather['Fecha_Diaria']).dt.normalize()

# 3. Combinar ambos DataFrames por la fecha
print("Combinando los datasets...")
df_final = pd.merge(df_gen, df_weather, left_on='Fecha', right_on='Fecha_Diaria', how='inner')

# Eliminar columna duplicada de fecha
df_final.drop(columns=['Fecha_Diaria'], inplace=True)

# 4. Crear columnas de Dia y Mes ("crear dataset dia y mes")
print("Extrayendo el día y mes de la fecha...")
df_final.insert(1, 'Mes', df_final['Fecha'].dt.month)
df_final.insert(2, 'Dia', df_final['Fecha'].dt.day)

# 5. Exportar el archivo final
print(f"Exportando a {output_file}...")
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"=== Proceso finalizado. Total de registros: {len(df_final)} ===")
print(f"Las columnas finales son: {list(df_final.columns)}")
