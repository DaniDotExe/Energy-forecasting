import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar el dataset generado previamente
df = pd.read_csv('dataset_clinico_taller.csv')

print("--- INFORMACIÓN ANTES DE LA LIMPIEZA ---")
df.info()
print("\n--- DATOS NULOS INICIALES ---")
print(df.isnull().sum())
print(f"Total general de datos nulos en todo el dataset: {df.isnull().sum().sum()}")
print("\n")

# ==========================================
# FASE 1: LIMPIEZA Y TRANSFORMACIÓN DE DATOS
# ==========================================

# 1.1 Validación de Privacidad (Anonimización)
df = df.drop(columns=['Nombre_Paciente', 'Documento_Identidad'], errors='ignore')

# Problema 1: Tipo de dato incorrecto (String con " bpm")
df['Frecuencia_Cardiaca'] = df['Frecuencia_Cardiaca'].astype(str).str.replace(' bpm', '')
df['Frecuencia_Cardiaca'] = pd.to_numeric(df['Frecuencia_Cardiaca'])

# Problema 2: Anomalías de sensores en Presión Arterial (-45 y 999)
df_limpio = df[(df['Presion_Arterial_Sistolica'] >= 60) & (df['Presion_Arterial_Sistolica'] <= 250)].copy()

# Problema 3: Datos Faltantes (NaN) en Nivel_Vitamina_D y Peso_kg
# Imputación con la media para tratar los valores perdidos
""" df_limpio['Nivel_Vitamina_D'] = df_limpio['Nivel_Vitamina_D'].fillna(df_limpio['Nivel_Vitamina_D'].mean())
df_limpio['Peso_kg'] = df_limpio['Peso_kg'].fillna(df_limpio['Peso_kg'].mean())
 """
print(f"Registros originales: {len(df)} | Registros limpios (sin anomalías y con imputación): {len(df_limpio)}\n")

print("--- RESULTADOS FASE 3: ANÁLISIS DE ASIMETRÍAS Y CURTOSIS ---")
if 'Edad' in df_limpio.columns:
    print(f"Asimetría Edad: {df_limpio['Edad'].skew():.2f} (Asimetría negativa)")
    print(f"Media de Edad: {df_limpio['Edad'].mean():.2f} | Mediana de Edad: {df_limpio['Edad'].median():.2f}")
if 'Nivel_Trigliceridos' in df_limpio.columns:
    print(f"Asimetría Triglicéridos: {df_limpio['Nivel_Trigliceridos'].skew():.2f} (Asimetría positiva)")
if 'Nivel_Vitamina_D' in df_limpio.columns:
    print(f"Curtosis Vitamina D: {df_limpio['Nivel_Vitamina_D'].kurt():.2f} (Platicúrtica, plana)")
if 'Nivel_Glucosa' in df_limpio.columns:
    print(f"Curtosis Glucosa: {df_limpio['Nivel_Glucosa'].kurt():.2f} (Leptocúrtica, con colas pesadas)")

if 'Horas_Actividad_Semana' in df_limpio.columns and 'IMC' in df_limpio.columns and 'Peso_kg' in df_limpio.columns and 'Colesterol_Total' in df_limpio.columns:
    corr_neg = df_limpio['Horas_Actividad_Semana'].corr(df_limpio['IMC'])
    corr_pos = df_limpio['Peso_kg'].corr(df_limpio['Colesterol_Total'])
    print("\n--- RESULTADOS FASE 3: CORRELACIONES ---")
    print(f"Correlación (Actividad vs IMC): {corr_neg:.2f} (Correlación negativa)")
    print(f"Correlación (Peso vs Colesterol): {corr_pos:.2f} (Correlación positiva)\n")

# ==========================================
# FASE 2: VISUALIZACIÓN EXPLORATORIA (Solo Matplotlib)
# ==========================================

# --- 2.1 Análisis Descriptivo: Formas de Distribución ---
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle('Análisis Descriptivo: Distribuciones Clínicas', fontsize=16, fontweight='bold')

# Gráfica 1: Edad (Asimetría Izquierda)
axes[0, 0].hist(df_limpio['Edad'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribución de Edad (Asimetría Izquierda o Negativa)')
axes[0, 0].set_xlabel('Edad (Años)')
axes[0, 0].set_ylabel('Frecuencia')

# Gráfica 2: Triglicéridos (Asimetría Derecha)
axes[0, 1].hist(df_limpio['Nivel_Trigliceridos'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Nivel de Triglicéridos (Asimetría Derecha o Positiva)')
axes[0, 1].set_xlabel('Triglicéridos (mg/dL)')
axes[0, 1].set_ylabel('Frecuencia')

# Gráfica 3: Vitamina D (Platicúrtica)
axes[1, 0].hist(df_limpio['Nivel_Vitamina_D'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Nivel de Vitamina D (Platicúrtica)')
axes[1, 0].set_xlabel('Vitamina D (ng/mL)')
axes[1, 0].set_ylabel('Frecuencia')

# Gráfica 4: Glucosa (Leptocúrtica)
axes[1, 1].hist(df_limpio['Nivel_Glucosa'], bins=40, color='plum', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Nivel de Glucosa (Leptocúrtica)')
axes[1, 1].set_xlabel('Glucosa (mg/dL)')
axes[1, 1].set_ylabel('Frecuencia')

# Gráfica 5: Presión Arterial limpia (Diagrama de caja)
box = axes[2, 0].boxplot(df_limpio['Presion_Arterial_Sistolica'], vert=False, patch_artist=True)
for patch in box['boxes']:
    patch.set_facecolor('gold')
axes[2, 0].set_title('Presión Arterial Sistólica (Sensores Limpios)')
axes[2, 0].set_xlabel('Presión Arterial (mmHg)')
axes[2, 0].set_yticks([])

# Ocultar el espacio sobrante (Gráfica 6)
axes[2, 1].axis('off')

plt.tight_layout()
plt.savefig('distribuciones_clinicas.png', dpi=300)
# plt.show() # Deshabilitado porque el entorno no es interactivo


# --- 2.2 Análisis Multivariable: Correlaciones ---
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Análisis de Correlaciones Bivariadas', fontsize=16, fontweight='bold')

# Función auxiliar para calcular la línea de tendencia (Regresión Lineal Simple)
def graficar_tendencia(ax, x, y, color_linea):
    # Filtramos valores nulos (NaN) para que polyfit no genere error de SVD
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() > 1:
        # np.polyfit calcula la pendiente (m) y el intercepto (b) de grado 1
        m, b = np.polyfit(x[mask], y[mask], 1) 
        ax.plot(x[mask], m*x[mask] + b, color=color_linea, linewidth=2)

# Correlación Negativa
x1, y1 = df_limpio['Horas_Actividad_Semana'], df_limpio['IMC']
axes2[0].scatter(x1, y1, alpha=0.4, color='gray')
graficar_tendencia(axes2[0], x1, y1, 'red') # Trazamos la línea
axes2[0].set_title('Correlación Negativa: Actividad vs IMC')
axes2[0].set_xlabel('Horas de Actividad por Semana')
axes2[0].set_ylabel('IMC')

# Correlación Positiva
x2, y2 = df_limpio['Peso_kg'], df_limpio['Colesterol_Total']
axes2[1].scatter(x2, y2, alpha=0.4, color='gray')
graficar_tendencia(axes2[1], x2, y2, 'green') # Trazamos la línea
axes2[1].set_title('Correlación Positiva: Peso vs Colesterol')
axes2[1].set_xlabel('Peso (kg)')
axes2[1].set_ylabel('Colesterol Total (mg/dL)')

plt.tight_layout()
plt.savefig('correlaciones_bivariadas.png', dpi=300)
# plt.show() # Deshabilitado porque el entorno no es interactivo


# --- 2.3 El Mapa de Calor (Heatmap) Construido Manualmente ---
# Calculamos la matriz de correlación
cols_numericas = df_limpio.select_dtypes(include=['float64', 'int64', 'int32']).columns
matriz_corr = df_limpio[cols_numericas].corr()

fig3, ax3 = plt.subplots(figsize=(10, 8))

# Usamos imshow para mostrar la matriz como una imagen con colores
cax = ax3.imshow(matriz_corr, cmap='coolwarm', vmin=-1, vmax=1)

# Añadimos la barra de colores a la derecha
fig3.colorbar(cax)

# Configuramos las etiquetas de los ejes X e Y
ax3.set_xticks(np.arange(len(cols_numericas)))
ax3.set_yticks(np.arange(len(cols_numericas)))
ax3.set_xticklabels(cols_numericas, rotation=45, ha='right')
ax3.set_yticklabels(cols_numericas)

# Bucle para escribir los valores numéricos dentro de cada cuadro
for i in range(len(cols_numericas)):
    for j in range(len(cols_numericas)):
        valor = matriz_corr.iloc[i, j]
        # Cambiamos el color del texto a blanco o negro dependiendo del fondo para que se lea bien
        color_texto = "white" if abs(valor) > 0.5 else "black"
        ax3.text(j, i, f"{valor:.2f}", ha="center", va="center", color=color_texto)

plt.title('Mapa de Calor General de Correlaciones', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300)
# plt.show() # Deshabilitado porque el entorno no es interactivo