#!/usr/bin/env python3
import pandas as pd
import os

base_dir = r"d:\Software\Energy-forecasting"
generation_file = os.path.join(base_dir, "data_2019_2023_kwH_xm.csv")
weather_file = os.path.join(base_dir, "clima_diario_el_paso_diurno.csv")
daily_output_file = os.path.join(base_dir, "daily_data.csv")
monthly_output_file = os.path.join(base_dir, "monthly_data.csv")

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

# 4. Crear columnas de Año, Mes y Dia
print("Extrayendo el año, mes y día de la fecha...")
df_final.insert(1, 'Año', df_final['Fecha'].dt.year)
df_final.insert(2, 'Mes', df_final['Fecha'].dt.month)
df_final.insert(3, 'Dia', df_final['Fecha'].dt.day)

# Calcular el promedio de generación por día
df_final['Promedio_Generacion'] = df_final[horas_cols].mean(axis=1)

# 5. Exportar el archivo diario
print(f"Exportando datos diarios a {daily_output_file}...")
df_final.to_csv(daily_output_file, index=False, encoding='utf-8-sig')

# 6. Crear y exportar el dataset mensual (promedio)
print("Calculando promedio mensual...")
numeric_cols = horas_cols + ['Promedio_Generacion', 'Temperatura_Tt', 'Viento_Wt', 'Irradiancia_It']
df_monthly = df_final.groupby(['Año', 'Mes', 'Recurso'])[numeric_cols].mean().reset_index()

print(f"Exportando datos mensuales a {monthly_output_file}...")
df_monthly.to_csv(monthly_output_file, index=False, encoding='utf-8-sig')

print(f"=== Proceso finalizado. Registros diarios: {len(df_final)}, mensuales: {len(df_monthly)} ===")
