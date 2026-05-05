#!/usr/bin/env python3
"""
API_OpenMeteo_Solar.py

Este script descarga el histórico climático de la planta solar "El Paso" (Cesar, Colombia)
utilizando la API pública Historical Weather API de Open-Meteo (sin API key).
Se obtienen los datos de irradiancia, temperatura, humedad y viento filtrados por horas diurnas (6am - 5pm).

Requerimientos:
    - requests
    - pandas
"""

import requests
import pandas as pd
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
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
TIMEZONE = "America/Bogota"

# Variables a extraer de la API (hourly)
HOURLY_VARIABLES = [
    "shortwave_radiation",   # Irradiancia
    "temperature_2m",        # Temperatura
    "relative_humidity_2m",  # Humedad
    "wind_speed_10m"         # Viento
]

# URL base de la API Historical Weather de Open-Meteo
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Nombre del archivo de salida
# Se guarda en la carpeta 'data' del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "openmeteo-horario.csv")


def fetch_weather_data() -> dict:
    """
    Realiza la petición GET a la API de Open-Meteo para obtener los datos horarios.
    Retorna los datos en formato JSON.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE
    }
    
    logging.info(f"Realizando petición a Open-Meteo para coordenadas ({LATITUDE}, {LONGITUDE}) desde {START_DATE} hasta {END_DATE}...")
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status() # Lanza excepción si el código de estado no es 200 OK
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error al conectar con la API: {e}")
        sys.exit(1)


def process_and_filter_data(json_data: dict) -> pd.DataFrame:
    """
    Convierte el JSON a un DataFrame de Pandas.
    """
    logging.info("Convirtiendo los datos JSON a DataFrame de Pandas...")
    
    # Extraer los datos horarios
    hourly_data = json_data.get("hourly", {})
    if not hourly_data:
        logging.error("No se encontraron datos horarios ('hourly') en la respuesta de la API.")
        sys.exit(1)
        
    df = pd.DataFrame({
        "time": hourly_data.get("time", []),
        "shortwave_radiation": hourly_data.get("shortwave_radiation", []),
        "temperature_2m": hourly_data.get("temperature_2m", []),
        "relative_humidity_2m": hourly_data.get("relative_humidity_2m", []),
        "wind_speed_10m": hourly_data.get("wind_speed_10m", [])
    })
    
    # Convertir 'time' a datetime para filtrar
    df["time"] = pd.to_datetime(df["time"])
    
    logging.info("Filtrando datos para el rango horario de 06:00 a 17:00 (6am - 5pm)...")
    df_filtered = df[(df["time"].dt.hour >= 6) & (df["time"].dt.hour <= 17)].copy()
    
    return df_filtered


def format_and_export(df: pd.DataFrame, output_path: str):
    """
    Renombra las columnas al formato exigido y exporta a CSV sin el índice.
    """
    logging.info("Renombrando columnas al formato de salida especificado...")
    
    # Renombrar columnas
    column_mapping = {
        "time": "Fecha_Hora",
        "shortwave_radiation": "Irradiancia_It",
        "temperature_2m": "Temperatura_Tt",
        "relative_humidity_2m": "Humedad_Ht",
        "wind_speed_10m": "Viento_Wt"
    }
    df = df.rename(columns=column_mapping)
    
    logging.info(f"Exportando resultados a: {output_path}")
    
    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Exportar a CSV excluyendo el índice
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    logging.info("Exportación completada con éxito.")


def main():
    """
    Función principal que orquesta la ejecución del script.
    """
    logging.info("=== Inicio del procesamiento de datos climáticos (Horario) ===")
    
    # 1. Extraer datos (API GET)
    data = fetch_weather_data()
    
    # 2. Procesar y filtrar datos (6am - 5pm)
    df_filtered = process_and_filter_data(data)
    
    # 3. Formatear y exportar a CSV
    format_and_export(df_filtered, OUTPUT_FILE)
    
    logging.info(f"=== Proceso finalizado. Archivo generado: {OUTPUT_FILE} ===")


if __name__ == "__main__":
    main()
