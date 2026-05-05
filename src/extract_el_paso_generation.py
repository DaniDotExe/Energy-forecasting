#!/usr/bin/env python3
import pandas as pd
import os

base_dir = r"d:\Software\Energy-forecasting"
years = [2019, 2020, 2021, 2022, 2023]

all_data = []

print("=== Starting data extraction for EL PASO (Solar) ===")

for year in years:
    file_path = os.path.join(base_dir, "data", f"Generacion_(kWh)_{year}.xlsx")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
        
    print(f"Processing year {year} from {file_path}...")
    
    try:
        # Los encabezados de las tablas de XM están en la fila 3 (índice 2)
        df = pd.read_excel(file_path, header=2)
        
        # Guardar las columnas originales para no perder los formatos de horas
        original_cols = df.columns
        
        # Limpiar temporalmente los nombres de las columnas para buscar
        cleaned_cols = [str(c).strip().upper() for c in df.columns]
        df.columns = cleaned_cols
        
        # Encontrar las columnas relevantes ("Recurso" y "Combustible")
        recurso_col = next((c for c in cleaned_cols if 'RECURSO' in c), None)
        combustible_col = next((c for c in cleaned_cols if 'COMBUSTIBLE' in c), None)
        
        if recurso_col and combustible_col:
            # Filtrar por Recurso = "EL PASO" y Combustible que contenga "SOLAR"
            mask = (
                df[recurso_col].astype(str).str.strip().str.upper() == 'EL PASO'
            ) & (
                df[combustible_col].astype(str).str.strip().str.upper().str.contains('SOLAR')
            )
            
            df_filtered = df[mask].copy()
            
            if not df_filtered.empty:
                print(f"  -> Found {len(df_filtered)} records for EL PASO in {year}")
                # Restaurar los nombres de las columnas originales para exportar
                df_filtered.columns = original_cols
                all_data.append(df_filtered)
            else:
                print(f"  -> No solar generation records found for EL PASO in {year}")
        else:
            print(f"  -> Error: Could not find 'Recurso' or 'Combustible' columns in {year}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if all_data:
    print("Concatenating data from all years...")
    final_df = pd.concat(all_data, ignore_index=True)
    
    output_file = os.path.join(base_dir, "data", "data_2019_2023_kwH_xm.csv")
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"=== Successfully saved aggregated data to: {output_file} ===")
    print(f"Total rows extracted: {len(final_df)}")
else:
    print("=== No data was found for EL PASO across the specified years. ===")
