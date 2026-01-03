import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATASET = "Amazon.csv"

# --- CONFIGURACIÓN DE VISUALIZACIÓN EN PANDAS ---
pd.set_option('display.max_columns', None) # Muestra todas las columnas del DataFrame (sin truncarlas)
pd.set_option('display.width', 180) # Ajusta el ancho máximo de la salida en consola
pd.set_option('display.max_rows', 100) # Muestra hasta 100 filas como máximo
pd.set_option('display.float_format', '{:.3f}'.format) # Formato para números flotantes: 2 decimales

# --- CONFIGURACIÓN DE ESTILO PARA GRÁFICOS ---
plt.style.use('seaborn-v0_8-darkgrid') # Define un estilo visual oscuro con cuadrícula para matplotlib
#sns.set_palette("husl") # Define la paleta de colores para seaborn

# --- CARGA DEL DATASET ---
df = pd.read_csv(DATASET) # Lee el archivo CSV y guardarlo en un DataFrame

# --- INFORMACIÓN GENERAL DEL DATASET ---
print("\n1. DIMENSIONES - DATASET:")
# Muestra la forma del DataFrame (filas, columnas)
print(f"   Shape: {df.shape}")
print(f"   Total Registros: {df.shape[0]:,}")
print(f"   Total Variables: {df.shape[1]}")

# --- PRIMERAS FILAS DEL DATASET ---
print("\n2. PRIMERAS 5 FILAS:")
print(df.head())

# --- ÚLTIMAS FILAS DEL DATASET ---
print("\n3. ULTIMAS 5 FILAS:")
print(df.tail())

# --- TIPOS DE VARIABLES DEL DATASET ---
print(f"\n4. Tipos de Variables")
print(f"{df.info()}\n")

# --- REVISIÓN DE VALORES NULOS ---
print(f"\n5. Revisión de Valores Nulos")
missing_data = df.isnull().sum() # Suma cuantos valores nulos hay por columna
missing_percentage = (missing_data / len(df)) * 100 # Calcula el porcenta de valores nulos respecto al total de filas
missing_df = pd.DataFrame({
    'Cantidad Nulos': missing_data,
    'Porcentaje': missing_percentage
})
missing_df = missing_df[missing_df['Cantidad Nulos'] > 0] # Filtra columnas con valores faltantes
if len(missing_df) > 0:
    print(missing_df)
else:
    print(f"   Valores nulos: {len(missing_df)}")

# --- REVISIÓN DE VALORES DUPLICADOS ---
print("\n6. Revisión de registros duplicados:")
duplicates = df.duplicated().sum()
print(f"   Filas Duplicadas: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"   {duplicates} filas duplicadas eliminadas")

# --- ESTADÍSTICAS BÁSICAS ---
print("\n7. ESTADÍSTICAS BÁSICAS - Variables Numéricas:")
print(df.describe())

numericas = df.select_dtypes(include='number') # Seleccionamos solo las columnas numéricas
# Dibujar un boxplot por cada variable
for col in numericas.columns:
    plt.figure(figsize=(6,4))
    plt.boxplot(numericas[col], vert=True, patch_artist=True)
    plt.title(f"Boxplot de {col}")
    plt.ylabel(col)
    plt.show()

print("\n8. ESTADÍSTICAS BÁSICAS - Variables Categóricas:")
print(df.describe(include="object").T)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nTop 5 valores de {col}:")
    print(f"{df[col].value_counts().head()}")

# --- EXTRAER DATOS DE LA FECHA ---
print("\n9. CONVERSIÓN DE ORDERDATE A TIPO FECHA Y EXTRACCIÓN DE SUS COMPONENTES:")
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce') # En caso de existir fechas inválidas las convierte en NaT (missing).

df['OrderYear'] = df['OrderDate'].dt.year
df['OrderMonth'] = df['OrderDate'].dt.month
df['OrderDay'] = df['OrderDate'].dt.day # Dia del mes
df['OrderQuarter'] = df['OrderDate'].dt.quarter # Número del trimestre al que pertenece la fecha
print(f"\n{df.head()}")

# # Check for inconsistent data
# print("\n10. REVISIÓN DE INCONSISTENCIAS:")
# numerical_cols = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost', 'TotalAmount']
# for col in numerical_cols:
#     negative_count = (df[col] < 0).sum()
#     if negative_count > 0:
#         print(f"   Warning: {negative_count} negative values found in {col}")


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




