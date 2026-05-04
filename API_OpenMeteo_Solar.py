#!/usr/bin/env python3
"""
API_OpenMeteo_Solar.py

Este script descarga el histórico climático de la planta solar "El Paso" (Cesar, Colombia)
utilizando la API pública Historical Weather API de Open-Meteo (sin API key).
Se filtran los datos para el rango de luz solar (6:00 a.m. a 5:00 p.m.) y se agrupan
mensualmente para su uso en un modelo de Red Neuronal Recurrente.

Requerimientos:
    - requests
    - pandas
"""

import requests
import pandas as pd
import logging
import sys

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- CONFIGURACIÓN DE PARÁMETROS ---
LATITUDE = 9.6667
LONGITUDE = -73.7500
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
TIMEZONE = "America/Bogota"

# Variables a extraer de la API (hourly)
HOURLY_VARIABLES = [
    "temperature_2m",      # Temperatura
    "wind_speed_10m",      # Velocidad del viento
    "shortwave_radiation"  # Irradiancia
]

# URL base de la API Historical Weather de Open-Meteo
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Nombre del archivo de salida
OUTPUT_FILE = "clima_mensual_el_paso_diurno.csv"


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
    Convierte la columna de tiempo a datetime.
    Filtra para conservar únicamente las horas del día entre las 06:00 y las 18:00 (inclusive).
    """
    logging.info("Convirtiendo los datos JSON a DataFrame de Pandas...")
    
    # Extraer los datos horarios
    hourly_data = json_data.get("hourly", {})
    if not hourly_data:
        logging.error("No se encontraron datos horarios ('hourly') en la respuesta de la API.")
        sys.exit(1)
        
    # Crear DataFrame
    df = pd.DataFrame({
        "time": hourly_data.get("time", []),
        "temperature_2m": hourly_data.get("temperature_2m", []),
        "wind_speed_10m": hourly_data.get("wind_speed_10m", []),
        "shortwave_radiation": hourly_data.get("shortwave_radiation", [])
    })
    
    # Convertir 'time' a formato datetime de Pandas
    df["time"] = pd.to_datetime(df["time"])
    
    logging.info("Filtrando datos para el rango horario de luz solar (06:00 - 17:00)...")
    
    # Filtrar entre las 06:00 y las 17:00 inclusive
    # (dt.hour extrae la hora en formato 24h, 0 a 23)
    df_diurno = df[(df["time"].dt.hour >= 6) & (df["time"].dt.hour <= 17)].copy()
    
    return df_diurno


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una columna 'Año-Mes' y agrupa los datos calculando el promedio mensual
    de las variables climáticas.
    """
    logging.info("Creando columna Año-Mes y agrupando promedios mensuales...")
    
    # Crear columna "Año-Mes" (ej. "2019-01") basándonos en la fecha
    df["Año-Mes"] = df["time"].dt.to_period("M")
    
    # Agrupar por "Año-Mes" y calcular la media de las variables numéricas
    df_monthly = df.groupby("Año-Mes")[["temperature_2m", "wind_speed_10m", "shortwave_radiation"]].mean().reset_index()
    
    # Convertir el tipo Period a string con formato YYYY-MM para que sea fácil de exportar
    df_monthly["Año-Mes"] = df_monthly["Año-Mes"].dt.strftime("%Y-%m")
    
    return df_monthly


def format_and_export(df: pd.DataFrame, output_path: str):
    """
    Renombra las columnas al formato exigido y exporta a CSV sin el índice.
    """
    logging.info("Renombrando columnas al formato de salida especificado...")
    
    # Renombrar columnas
    column_mapping = {
        "Año-Mes": "Fecha_Mensual",
        "temperature_2m": "Temperatura_Tt",
        "wind_speed_10m": "Viento_Wt",
        "shortwave_radiation": "Irradiancia_It"
    }
    df = df.rename(columns=column_mapping)
    
    logging.info(f"Exportando resultados a: {output_path}")
    
    # Exportar a CSV excluyendo el índice
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    logging.info("Exportación completada con éxito.")


def main():
    """
    Función principal que orquesta la ejecución del script.
    """
    logging.info("=== Inicio del procesamiento de datos climáticos ===")
    
    # 1. Extraer datos (API GET)
    data = fetch_weather_data()
    
    # 2. Procesar y filtrar (Horas diurnas)
    df_diurno = process_and_filter_data(data)
    
    # 3. Agrupar mensualmente
    df_mensual = aggregate_monthly(df_diurno)
    
    # 4. Formatear y exportar a CSV
    format_and_export(df_mensual, OUTPUT_FILE)
    
    logging.info("=== Proceso finalizado. Los datos están listos para la Red Neuronal Recurrente. ===")


if __name__ == "__main__":
    main()
