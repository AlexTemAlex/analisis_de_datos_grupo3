import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATASET = "Amazon.csv"

df = pd.read_csv(DATASET, decimal='.', skipinitialspace=True)

# Limpiar nombres de columna:
# - quitar espacios
# - reemplazar caracteres especiales por _
# - pasar todo a minúsculas (opcional)
df.columns = (
    df.columns.str.strip()                # Quita espacios
              .str.replace(r'[^\w]', '_', regex=True)  # Reemplaza todo lo que no sea letra/número por _
              .str.lower()                # Minusculas opcional
)

# Limpiar los datos de strings
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Forzar columnas numéricas
numeric_cols = ['quantity', 'unitprice', 'discount', 'tax', 'shippingcost', 'totalamount']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Guardar CSV final
df.to_csv("Amazon_Discover_Renamed.csv", index=False, sep=',', decimal='.')









# pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
# pd.set_option('display.width', 180)         # Ajusta el ancho de la salida
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.float_format', '{:.2f}'.format)

# # Set style for plots
# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set_palette("husl")

# # Load the dataset
# df = pd.read_csv(DATASET)

# print("\n1. DATASET DIMENSIONS:")
# print(f"   Shape: {df.shape}")
# print(f"   Total Records: {df.shape[0]:,}")
# print(f"   Total Features: {df.shape[1]}")

# # Display first few rows
# print("\n2. FIRST 5 ROWS:")
# print(df.head())

# # Display last few rows
# print("\n3. LAST 5 ROWS:")
# print(df.tail())

# dtypes = {
#     'year': 'Int16',
#     'month': 'Int16',
#     'day': 'Int16',
#     'weekofyear': 'Int16',
#     'weekday': 'Int16',
#     'is_weekend': 'category',
#     'is_holiday': 'category',
#     'temperature': 'float32',
#     'rain_mm': 'float32',
#     'store_id': 'category',
#     'country': 'category',
#     'city': 'category',
#     'channel': 'category',
#     'sku_id': 'category',
#     'sku_name': 'category',
#     'category': 'category',
#     'subcategory': 'category',
#     'brand': 'category',
#     'units_sold': 'Int16',
#     'list_price': 'float32',
#     'discount_pct': 'float32',
#     'promo_flag': 'category',
#     'gross_sales': 'float32',
#     'net_sales': 'float32',
#     'stock_on_hand': 'Int16',
#     'stock_out_flag': 'category',
#     'lead_time_days': 'Int16',
#     'supplier_id': 'category',
#     'purchase_cost': 'float32',
#     'margin_pct': 'float32'
# }

# df = pd.read_csv(DATASET,
#                 usecols=list(dtypes.keys()),  # solo columnas de dtypes
#                 dtype=dtypes)
#                 # nrows=1000) 

# print(f"*****Data Set*****\n{df}\n")
# print(f"*****Tipos de Variables Definidas*****")
# print(f"{df.info()}\n")

# print(f"*****Promedio de Valores Nulos*****\n{df.isnull().mean().sort_values(ascending=False)}\n")

# print(f"*****Cantidad de Valores Duplicados*****\n{df.duplicated(subset=['year','month','day','store_id','sku_id']).sum()}\n")


# #Especificacion de variables numericas y categoricas
# num_real = [
#     'temperature', 'rain_mm', 'units_sold',
#     'list_price', 'discount_pct', 'gross_sales', 'net_sales',
#     'stock_on_hand', 'lead_time_days', 'purchase_cost',
#     'margin_pct'
    
# ]

# flags = [
#     'is_weekend', 'is_holiday',
#     'promo_flag', 'stock_out_flag'
# ]

# categ = [
#     'year', 'month', 'day', 'weekofyear', 'weekday',
#     'store_id', 'country', 'city', 'channel', 'sku_id',
#     'sku_name', 'category', 'subcategory', 'brand', 'supplier_id'
# ]

# print(f"*****Etadisticas Básicas - Variables Numéricas*****\n{df[num_real].describe().T}\n")
# print(f"*****Etadisticas Básicas - Variables Numéricas Binarias (Codigo Disyuntivo)*****\n{df[flags].describe(include="category").T}\n")
# print(f"*****Etadisticas Básicas - Variables Categóricas***** \n{df[categ].describe(include="category").T}\n")

# print(f"*****Centrar y Reducir*****")
# df_num = df[num_real].dropna() # Eliminar Valores Nulos
# scaler = StandardScaler()

# df_scaled = pd.DataFrame(
#     scaler.fit_transform(df_num),
#     columns=num_real,
#     index=df_num.index
# )

# print(f"\n{df_scaled.describe().loc[['mean','std']]}\n")

# #Boxplot y Detección de Outlier
# # Deteccion de outliers
# outliers_dict = {}

# for col in df_scaled.columns:
#     Q1 = df_scaled[col].quantile(0.25)
#     Q3 = df_scaled[col].quantile(0.75)
#     IQR = Q3 - Q1
#     outliers = df_scaled[(df_scaled[col] < Q1 - 1.5 * IQR) | (df_scaled[col] > Q3 + 1.5 * IQR)][col]
#     outliers_dict[col] = outliers.values  # Guardamos los valores outliers

# # Ejemplo: ver cuántos outliers hay por variable
# print(f"*****Conteo de Outliers*****")
# for var, out in outliers_dict.items():
#     print(f"{var}: {len(out)} outliers")

# print(f"*****Boxplot*****")
# # plt.figure(figsize=(15, 10)) # Crear figura

# # sns.boxplot(data=df_scaled, orient='h', palette="Set2") # Boxplot horizontal de todas las variables

# # plt.title("Boxplots de Variables Numéricas (Centradas y Reducidas) con Outliers")
# # plt.xlabel("Valor Estandarizado")
# # plt.ylabel("Variables")
# # plt.show()

# print(f"*****Grafico de Dispersion*****")
# # target = 'net_sales' # Variable de referencia

# # # Recorremos todas las columnas numéricas menos la variable de referencia
# # for col in df_scaled.columns:
# #     if col == target:
# #         continue  # No queremos graficar la variable de referencia contra sí misma
    
# #     plt.figure(figsize=(8, 5))
# #     sns.scatterplot(
# #         x=df_scaled[col],
# #         y=df_scaled[target],
# #         alpha=0.6,
# #         color='dodgerblue'
# #     )
# #     plt.title(f"Gráfico de Dispersión: {col} vs {target}")
# #     plt.xlabel(f"{col} (Estandarizado)")
# #     plt.ylabel(f"{target} (Estandarizado)")
# #     plt.show()
    

# # Calcular la matriz de correlación
# corr_matrix = df_scaled.corr()

# plt.figure(figsize=(12,8))

# # Graficar heatmap
# sns.heatmap(
#     corr_matrix, 
#     annot=True,       # Muestra los valores de correlación en cada celda
#     fmt=".3f",        # Formato con 2 decimales
#     cmap="vlag",  # Paleta de colores
#     cbar=True,        # Mostrar barra de colores
#     square=True,       # Cuadrado para cada celda
#     linewidths=0.5
# )

# plt.title("Matriz de Correlación - Variables Numéricas")
# plt.show()

# #ACP
# # Asumiendo que df_scaled tiene todas las variables numéricas centradas y reducidas
# n_components = df_scaled.shape[1]  # Número de componentes igual al número de variables
# pca = PCA(n_components=n_components)
# principal_components = pca.fit_transform(df_scaled)

# # Convertir a DataFrame para manejarlo más fácilmente
# df_pca = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

# explained_variance = pca.explained_variance_ratio_
# cum_var = explained_variance.cumsum()

# # Mostrar la varianza de cada componente
# for i, var in enumerate(explained_variance):
#     print(f"PC{i+1}: {var:.4f} ({cum_var[i]:.4f} acumulada)")

# # Gráfico de varianza acumulada
# plt.figure(figsize=(8,5))
# plt.plot(range(1, n_components+1), cum_var, marker='o', linestyle='--', color='b')
# plt.title("Varianza Acumulada (Inercia) - PCA")
# plt.xlabel("Número de Componentes Principales")
# plt.ylabel("Varianza Acumulada")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,6))
# plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.6, color='dodgerblue')
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Proyección en los dos primeros Componentes Principales")
# plt.grid(True)
# plt.show()
