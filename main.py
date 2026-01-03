import matplotlib
# Usar backend no interactivo para evitar problemas con Tkinter
matplotlib.use('Agg')  # 'Agg' es un backend que no necesita GUI

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATASET = "Amazon.csv"

# --- CONFIGURACIÓN DE VISUALIZACIÓN EN PANDAS ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.3f}'.format)

# --- CONFIGURACIÓN DE ESTILO PARA GRÁFICOS ---
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- CARGA DEL DATASET ---
df = pd.read_csv(DATASET)

# --- INFORMACIÓN GENERAL DEL DATASET ---
print("\n1. DIMENSIONES - DATASET:")
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
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Cantidad Nulos': missing_data,
    'Porcentaje': missing_percentage
})
missing_df = missing_df[missing_df['Cantidad Nulos'] > 0]
if len(missing_df) > 0:
    print(missing_df)
else:
    print("   No se encontraron valores nulos!")

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

print("\n8. ESTADÍSTICAS BÁSICAS - Variables Categóricas:")
print(df.describe(include="object").T)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nTop 5 valores de {col}:")
    print(f"{df[col].value_counts().head()}")

# --- EXTRAER DATOS DE LA FECHA ---
print("\n9. CONVERSIÓN DE ORDERDATE A TIPO FECHA Y EXTRACCIÓN DE SUS COMPONENTES:")
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
df['OrderYear'] = df['OrderDate'].dt.year
df['OrderMonth'] = df['OrderDate'].dt.month
df['OrderDay'] = df['OrderDate'].dt.day
df['OrderQuarter'] = df['OrderDate'].dt.quarter
print(f"\n{df.head()}")

# ============================================================================
# PUNTO 10: ANÁLISIS DE CORRELACIÓN (Basado en tu código comentado)
# ============================================================================

print("\n" + "="*80)
print("10. ANÁLISIS DE CORRELACIÓN")
print("="*80)

# Identificar variables numéricas para análisis
numerical_vars = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost', 'TotalAmount']

# Asegurarse de que todas las variables existan en el DataFrame
numerical_vars = [var for var in numerical_vars if var in df.columns]

print(f"Variables numéricas para análisis de correlación: {numerical_vars}")

# Preparar datos numéricos (eliminar NaN)
df_num = df[numerical_vars].dropna()

# Estandarizar los datos (centrar y reducir)
print("\n10.1 ESTANDARIZACIÓN DE DATOS (Centrar y Reducir):")
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_num),
    columns=numerical_vars,
    index=df_num.index
)

print("Estadísticas después de estandarización:")
print(df_scaled.describe().loc[['mean', 'std']])

# Calcular matriz de correlación
print("\n10.2 MATRIZ DE CORRELACIÓN:")
corr_matrix = df_scaled.corr()
print(corr_matrix)

# Visualizar matriz de correlación con heatmap
print("\n10.3 VISUALIZACIÓN DE MATRIZ DE CORRELACIÓN (Heatmap):")
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True,
    fmt=".3f",
    cmap="vlag",
    cbar=True,
    square=True,
    linewidths=0.5,
    vmin=-1, vmax=1,
    center=0
)
plt.title("Matriz de Correlación - Variables Numéricas", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')  # CAMBIADO: plt.show() por plt.savefig()
print("   Gráfico guardado como 'matriz_correlacion.png'")

# Análisis de correlaciones más fuertes
print("\n10.4 ANÁLISIS DE CORRELACIONES:")
threshold = 0.5  # Umbral para correlaciones fuertes

print(f"Correlaciones fuertes (|r| > {threshold}):")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            direction = "positiva" if corr_value > 0 else "negativa"
            print(f"  • {var1} ↔ {var2}: {corr_value:.3f} ({direction})")

# Correlación con TotalAmount (si existe)
if 'TotalAmount' in numerical_vars:
    print(f"\nCorrelación con TotalAmount (variable objetivo):")
    total_corr = corr_matrix['TotalAmount'].sort_values(ascending=False)
    for var, corr in total_corr.items():
        if var != 'TotalAmount':
            print(f"  • {var}: {corr:.3f}")

# ============================================================================
# PUNTO 11: ANÁLISIS DE COMPONENTES PRINCIPALES (ACP)
# ============================================================================

print("\n" + "="*80)
print("11. ANÁLISIS DE COMPONENTES PRINCIPALES (ACP)")
print("="*80)

# Usar los datos ya estandarizados (df_scaled) para el ACP
print("\n11.1 APLICACIÓN DE ANÁLISIS DE COMPONENTES PRINCIPALES:")

n_components = min(df_scaled.shape[1], df_scaled.shape[0])
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(df_scaled)

# Convertir a DataFrame para manejo más fácil
df_pca = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

# Varianza explicada
explained_variance = pca.explained_variance_ratio_
cum_var = explained_variance.cumsum()

print("\n11.2 VARIANZA EXPLICADA POR CADA COMPONENTE:")
for i, (var, cum) in enumerate(zip(explained_variance, cum_var), 1):
    print(f"  PC{i}: {var:.4f} ({var*100:.1f}%) - Acumulada: {cum:.4f} ({cum*100:.1f}%)")

# Componentes necesarios para explicar diferentes porcentajes de varianza
print("\n11.3 COMPONENTES NECESARIOS PARA EXPLICAR X% DE VARIANZA:")
thresholds = [0.7, 0.8, 0.9, 0.95]
for threshold in thresholds:
    n_components_needed = np.where(cum_var >= threshold)[0][0] + 1
    print(f"  • {threshold*100:.0f}% de varianza: {n_components_needed} componentes")

# Visualización 1: Varianza acumulada (Scree plot)
print("\n11.4 VISUALIZACIÓN - VARIANZA ACUMULADA:")
plt.figure(figsize=(10, 5))

# Gráfico de varianza acumulada
plt.subplot(1, 2, 1)
plt.plot(range(1, n_components+1), cum_var, marker='o', linestyle='--', color='b')
plt.title("Varianza Acumulada (Inercia) - PCA")
plt.xlabel("Número de Componentes Principales")
plt.ylabel("Varianza Acumulada")
plt.grid(True)

# Gráfico de varianza individual
plt.subplot(1, 2, 2)
plt.bar(range(1, n_components+1), explained_variance, alpha=0.7, color='skyblue')
plt.title("Varianza Individual por Componente")
plt.xlabel("Componente Principal")
plt.ylabel("Varianza Explicada")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('varianza_acp.png', dpi=300, bbox_inches='tight')  # CAMBIADO: plt.show() por plt.savefig()
print("   Gráfico guardado como 'varianza_acp.png'")

# Visualización 2: Proyección en los dos primeros componentes
if n_components >= 2:
    print("\n11.5 VISUALIZACIÓN - PROYECCIÓN EN PC1 Y PC2:")
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.6, color='dodgerblue')
    plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}% varianza)")
    plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}% varianza)")
    plt.title("Proyección en los dos primeros Componentes Principales")
    plt.grid(True, alpha=0.3)
    
    # Añadir líneas de referencia
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('proyeccion_pc1_pc2.png', dpi=300, bbox_inches='tight')  # CAMBIADO: plt.show() por plt.savefig()
    print("   Gráfico guardado como 'proyeccion_pc1_pc2.png'")

# Cargas factoriales (contribución de variables originales)
print("\n11.6 CARGAS FACTORIALES (Contribución de variables a cada componente):")
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=numerical_vars
)

# Mostrar las variables más importantes para los primeros 2-3 componentes
print("\nVariables más importantes para los primeros componentes:")
for i in range(min(3, n_components)):
    pc_name = f'PC{i+1}'
    print(f"\n• {pc_name} (explica {explained_variance[i]*100:.1f}% de varianza):")
    
    # Variables con mayor contribución (valor absoluto)
    contributions = loadings[pc_name].abs().sort_values(ascending=False).head(3)
    for var, loading in contributions.items():
        original_loading = loadings.loc[var, pc_name]
        direction = "positiva" if original_loading > 0 else "negativa"
        print(f"  - {var}: {original_loading:.3f} ({direction})")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)

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