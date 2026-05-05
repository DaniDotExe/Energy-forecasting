#!/usr/bin/env python3
"""
API_NASAPOWER.py

Este script descarga el histórico climático de la planta solar "El Paso" (Cesar, Colombia)
utilizando la API pública de NASA POWER (POWER Single Point Data Access).
Se obtienen los datos de irradiancia, temperatura, humedad y viento en formato horario,
filtrados por horas diurnas (6am - 5pm).

Requerimientos:
    - requests
    - pandas
    - numpy
"""

import requests
import pandas as pd
import numpy as np
import logging
import sys
import os

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- CONFIGURACIÓN DE PARÁMETROS ---
LATITUDE = 9.785041627157241
LONGITUDE = -73.7201783147516
START_DATE = "20190101"
END_DATE = "20231231"
TIMEZONE = "America/Bogota"

# Variables a extraer de la API NASA POWER (hourly)
# ALLSKY_SFC_SW_DWN: Irradiance (Wh/m^2)
# T2M: Temperature at 2 Meters (C)
# RH2M: Relative Humidity at 2 Meters (%)
# WS10M: Wind Speed at 10 Meters (m/s) -> Se convertirá a km/h
HOURLY_VARIABLES = "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M"

# URL base de la API de NASA POWER
API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Nombre del archivo de salida
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "nasapower-horario.csv")


def fetch_weather_data() -> dict:
    """
    Realiza la petición GET a la API de NASA POWER para obtener los datos horarios.
    Retorna los datos en formato JSON.
    """
    params = {
        "parameters": HOURLY_VARIABLES,
        "community": "RE",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
        "time-standard": "UTC"
    }
    
    logging.info(f"Realizando petición a NASA POWER para coordenadas ({LATITUDE}, {LONGITUDE}) desde {START_DATE} hasta {END_DATE}...")
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error al conectar con la API de NASA POWER: {e}")
        sys.exit(1)


def process_and_filter_data(json_data: dict) -> pd.DataFrame:
    """
    Convierte el JSON a un DataFrame de Pandas, ajusta la zona horaria,
    convierte unidades (viento m/s a km/h) y filtra por horas diurnas (6am - 5pm).
    """
    logging.info("Convirtiendo los datos JSON a DataFrame de Pandas...")
    
    parameters = json_data.get("properties", {}).get("parameter", {})
    if not parameters:
        logging.error("No se encontraron parámetros climáticos en la respuesta de la API.")
        sys.exit(1)
        
    # Crear DataFrame desde el diccionario
    df = pd.DataFrame(parameters)
    
    # El índice actual es la fecha en formato string "YYYYMMDDHH" (UTC)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "time_utc"}, inplace=True)
    
    # Convertir a datetime UTC
    df["time_utc"] = pd.to_datetime(df["time_utc"], format="%Y%m%d%H")
    # Asignar zona horaria UTC y convertir a America/Bogota
    df["time_local"] = df["time_utc"].dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    # Remover la información de zona horaria para guardar limpio
    df["time_local"] = df["time_local"].dt.tz_localize(None)
    
    # NASA POWER usa -999.0 para valores faltantes.
    df.replace(-999.0, np.nan, inplace=True)
    
    # Convertir la velocidad del viento de m/s a km/h
    df["WS10M"] = df["WS10M"] * 3.6
    
    logging.info("Filtrando datos para el rango horario de 06:00 a 17:00 (6am - 5pm) local...")
    df_filtered = df[(df["time_local"].dt.hour >= 6) & (df["time_local"].dt.hour <= 17)].copy()
    
    return df_filtered


def format_and_export(df: pd.DataFrame, output_path: str):
    """
    Renombra las columnas al formato exigido y exporta a CSV sin el índice.
    """
    logging.info("Renombrando columnas al formato de salida especificado...")
    
    # Renombrar columnas
    column_mapping = {
        "time_local": "Fecha_Hora",
        "ALLSKY_SFC_SW_DWN": "Irradiancia_It",
        "T2M": "Temperatura_Tt",
        "RH2M": "Humedad_Ht",
        "WS10M": "Viento_Wt"
    }
    
    # Seleccionar solo las columnas mapeadas
    df_out = df[list(column_mapping.keys())].rename(columns=column_mapping)
    
    logging.info(f"Exportando resultados a: {output_path}")
    
    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Exportar a CSV excluyendo el índice
    df_out.to_csv(output_path, index=False, encoding="utf-8")
    
    logging.info(f"Exportación completada con éxito. Registros exportados: {len(df_out)}")


def main():
    """
    Función principal que orquesta la ejecución del script.
    """
    logging.info("=== Inicio del procesamiento de datos climáticos de NASA POWER ===")
    
    # 1. Extraer datos (API GET)
    data = fetch_weather_data()
    
    # 2. Procesar y filtrar datos (6am - 5pm)
    df_filtered = process_and_filter_data(data)
    
    # 3. Formatear y exportar a CSV
    format_and_export(df_filtered, OUTPUT_FILE)
    
    logging.info(f"=== Proceso finalizado. Archivo generado: {os.path.basename(OUTPUT_FILE)} ===")


if __name__ == "__main__":
    main()
