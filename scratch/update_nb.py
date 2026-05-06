import json

with open('Pipeline_Execution.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if 'source' in cell and len(cell['source']) > 0:
        source_str = ''.join(cell['source'])
        
        # Update Cell 9: Data Loading (Fix indentation and ensure imports)
        if "df_monthly = pd.read_csv('data/monthly-data.csv')" in source_str:
            new_source = []
            for line in cell['source']:
                if "    # Convertir Mes a Datetime" in line:
                    new_source.append("# Convertir Mes a Datetime para indexación (opcional pero recomendado)\n")
                elif "    df_monthly['Mes'] = pd.to_datetime" in line:
                    new_source.append("df_monthly['Mes'] = pd.to_datetime(df_monthly['Mes'])\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print(f"Fixed indentation in cell {i}")

        # Update Cell 13: Grid Search (Ensure it's self-contained and reports errors)
        if "endog=df_monthly['kWh_Total']" in source_str:
            cell['source'] = [
                "import pandas as pd\n",
                "import os\n",
                "from itertools import product\n",
                "from statsmodels.tsa.statespace.sarimax import SARIMAX\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Asegurar que los datos estén cargados\n",
                "if 'df_monthly' not in locals():\n",
                "    if os.path.exists('data/monthly-data.csv'):\n",
                "        df_monthly = pd.read_csv('data/monthly-data.csv')\n",
                "    else:\n",
                "        print('Error: data/monthly-data.csv no encontrado. Por favor corre las celdas de descarga.')\n",
                "\n",
                "# Rango de valores a probar\n",
                "p_values = [1, 2, 3]\n",
                "d_values = [0]\n",
                "q_values = [1, 2]\n",
                "P_values = [1, 2]\n",
                "D_values = [1]\n",
                "Q_values = [1, 2]\n",
                "m = 12\n",
                "\n",
                "resultados = []\n",
                "combinaciones = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))\n",
                "print(f'Total combinaciones a probar: {len(combinaciones)}')\n",
                "\n",
                "for p, d, q, P, D, Q in combinaciones:\n",
                "    try:\n",
                "        modelo = SARIMAX(\n",
                "            endog=df_monthly['kWh_Total'],\n",
                "            exog=df_monthly[['Irradiancia_It_mean', 'Temperatura_Tt_mean', 'Viento_Wt_mean']],\n",
                "            order=(p, d, q),\n",
                "            seasonal_order=(P, D, Q, m),\n",
                "            enforce_stationarity=False,\n",
                "            enforce_invertibility=False\n",
                "        ).fit(disp=False)\n",
                "        \n",
                "        resultados.append({\n",
                "            'orden': f'({p},{d},{q})({P},{D},{Q})_{m}',\n",
                "            'AIC': modelo.aic,\n",
                "            'BIC': modelo.bic\n",
                "        })\n",
                "        print(f\"  {resultados[-1]['orden']} -> AIC: {modelo.aic:.1f}\")\n",
                "    except Exception as e:\n",
                "        # print(f'Error en {p,d,q,P,D,Q}: {e}')\n",
                "        pass\n",
                "\n",
                "if resultados:\n",
                "    resultados_df = pd.DataFrame(resultados).sort_values('AIC')\n",
                "    print('\\nTop 5 mejores modelos:')\n",
                "    print(resultados_df.head())\n",
                "else:\n",
                "    print('No se pudieron ajustar modelos. Revisa los datos exógenos.')\n"
            ]
            print(f"Updated grid search cell {i} to be self-contained")

with open('Pipeline_Execution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
