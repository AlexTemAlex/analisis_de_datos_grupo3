import matplotlib
# Usar backend no interactivo para evitar problemas con Tkinter
#matplotlib.use('Agg')  # 'Agg' es un backend que no necesita GUI

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
# -------------------------------------------------------------------------
# 1. CARGA DEL DATASET
# -------------------------------------------------------------------------
DATASET = "Amazon.csv"
df = pd.read_csv(DATASET)

# --- CONFIGURACIÓN DE VISUALIZACIÓN EN PANDAS ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.3f}'.format)

# --- CONFIGURACIÓN DE ESTILO PARA GRÁFICOS ---
plt.style.use('seaborn-v0_8-darkgrid') # Define un estilo visual oscuro con cuadrícula para matplotlib

# --- INFORMACIÓN GENERAL DEL DATASET ---
print("\n" + "="*80)
print("1. DIMENSIONES ORIGINALES - DATASET:")
print("="*80)
print(f"   Shape: {df.shape}")
print(f"   Total Registros: {df.shape[0]:,}")
print(f"   Total Variables: {df.shape[1]}")

print("\n1.1 Primeras 5 filas:")
print(df.head())
print("\n1.2 Ultimas 5 filas:")
print(df.tail())
print(f"\n1.3 Tipos de Variables")
print(f"{df.info()}\n")

# -------------------------------------------------------------------------
# 2. LIMPIEZA BÁSICA
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("2. Eliminar registros duplicados:")
print("="*80)
duplicates = df.duplicated().sum()
print(f"   Filas Duplicadas: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"   {duplicates} filas duplicadas eliminadas")

# Conversión de fecha
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')

# -------------------------------------------------------------------------
# 3. ELIMINACIÓN DE VARIABLES IRRELEVANTES
# -------------------------------------------------------------------------
drop_cols = [
    'OrderID',
    'OrderDate',
    'CustomerName',
    'ProductName',
]
print("\n" + "="*80)
print("3. Eliminar variables irrelevantes")
print("="*80)
print(f"3.1 Variables eliminadas: {drop_cols}")
df_model = df.drop(columns=drop_cols, errors='ignore')
print("3.2 Variables restantes:", df_model.columns.tolist())

# -------------------------------------------------------------------------
# 4. TRATAMIENTO DE VALORES NULOS
# -------------------------------------------------------------------------
print("\n" + "="*80)
print(f"4. Revisión de Valores Nulos")
print("="*80)
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
    print(f"   4.1 Valores nulos: {len(missing_df)}")

# -------------------------------------------------------------------------
# 6. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# -------------------------------------------------------------------------
# --- Convertir variables a codigo disyuntivo ---
print("\n" + "="*80)
print("6. CONVERSIÓN DE VARIABLES CATEGORICAS A CODIGO DISYUNTIVO:")
print("="*80)

categorical_vars = [ # Variables categóricas a codificar
    # 'Category',
    # 'PaymentMethod',
    # 'OrderStatus',
    # 'Country',
    # 'State',
    # 'City'
]

df_model = pd.get_dummies(
    df_model,
    columns=categorical_vars,
    drop_first=True,
    dtype=int
)

print("Dimensión con código disyuntivo:", df_model.shape)
print(df_model.head(n=3)) # Ejemplo de columnas creadas

# -------------------------------------------------------------------------
# 6. SELECCIÓN DE VARIABLES NUMÉRICAS
# -------------------------------------------------------------------------
numeric_cols = df_model.select_dtypes(include='number').columns.tolist()
X = df_model[numeric_cols]
print("Número de variables numéricas seleccionadas:", len(numeric_cols))

# -------------------------------------------------------------------------
# 7. ESTADÍSTICAS BÁSICAS
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("\n7.1 ESTADÍSTICAS BÁSICAS - Variables Numéricas:")
print("\n" + "="*80)
print(df.describe())

print("\n" + "="*80)
print("\n7.2 ESTADÍSTICAS BÁSICAS - Variables Categóricas:")
print("="*80)
print(df.describe(include="object").T)

# Estilo de los gráficos
sns.set_theme(style="whitegrid")

# Carpeta donde se guardarán las imágenes
output_folder = "graficos_categoricos"
os.makedirs(output_folder, exist_ok=True)

# Seleccionar columnas categóricas
categorical_cols = df.select_dtypes(include=['object']).columns

# Generar y guardar los gráficos
for col in categorical_cols:
    # Conteos de cada categoría
    counts = df[col].value_counts()
    
    # --- Top 5 categorías con mayor cantidad ---
    top5_high = counts.nlargest(5)
    plt.figure(figsize=(8,4))
    sns.barplot(
        x=top5_high.values,
        y=top5_high.index,
        color="skyblue" 
    )
    plt.title(f'Top 5 categorías más frecuentes de "{col}"')
    plt.xlabel('Cantidad')
    plt.ylabel(col)
    plt.tight_layout()
    file_path = os.path.join(output_folder, f"{col}_top5_alta.png")
    plt.savefig(file_path)
    plt.close()
    
    # --- Top 5 categorías con menor cantidad ---
    top5_low = counts.nsmallest(5)
    plt.figure(figsize=(8,4))
    sns.barplot(
        x=top5_low.values,
        y=top5_low.index,
        color="skyblue"   
    )
    plt.title(f'Top 5 categorías menos frecuentes de "{col}"')
    plt.xlabel('Cantidad')
    plt.ylabel(col)
    plt.tight_layout()
    file_path = os.path.join(output_folder, f"{col}_top5_baja.png")
    plt.savefig(file_path)
    plt.close()

print(f"Todos los gráficos (Top 5 altas y bajas) se guardaron en la carpeta '{output_folder}'")


# -------------------------------------------------------------------------
# 7. ESTANDARIZACIÓN (CENTRAR Y REDUCIR)
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("7. ESTANDARIZACIÓN DE DATOS (Centrar y Reducir)")
print("="*80)

scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index
)

print("Chequeo de estandarización:")
print(df_scaled.describe().loc[['mean', 'std']])

# -------------------------------------------------------------------------
# 8. ACP
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("8. PCA")
print("="*80)

pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(df_scaled)

explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

print(f"Número de componentes retenidos: {pca.n_components_}")
print(f"Varianza total explicada: {cum_var[-1]:.4f}")

pca_summary = pd.DataFrame({
    "Componente": [f"PC{i+1}" for i in range(len(explained_var))],
    "Varianza_Individual": explained_var,
    "Varianza_Acumulada": cum_var
})
print(pca_summary)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=df_scaled.columns
)

print("\nVariables más influyentes en PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).head(10))

print("\nVariables más influyentes en PC2:")
print(loadings["PC2"].abs().sort_values(ascending=False).head(10))
# Scree plot: varianza individual y acumulada
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(range(1, len(explained_var)+1), cum_var, marker='o', linestyle='--')
plt.title("Varianza Acumulada - PCA")
plt.xlabel("Número de Componentes")
plt.ylabel("Varianza Acumulada")
plt.grid(True)

plt.subplot(1,2,2)
plt.bar(range(1, len(explained_var)+1), explained_var, color='skyblue')
plt.title("Varianza Individual por Componente")
plt.xlabel("Componente")
plt.ylabel("Varianza Explicada")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("varianza_acp.png", dpi=300)
plt.close()

if pca.n_components_ >= 2:
    plt.figure(figsize=(7,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6)
    plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    plt.title("Proyección PCA (PC1 vs PC2)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("proyeccion_pca.png", dpi=300)
    plt.close()


print("\n" + "="*80)
print("9. K-MEANS SOBRE COMPONENTES PRINCIPALES")
print("="*80)

inertias = []
K = range(1, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    inertias.append(km.inertia_)

plt.figure(figsize=(7,5))
plt.plot(K, inertias, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del Codo")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_kmeans.png", dpi=300)
plt.close()
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster_kmeans"] = kmeans.fit_predict(X_pca)

centroides = kmeans.cluster_centers_

plt.figure(figsize=(8,6))
plt.scatter(
    X_pca[:,0], X_pca[:,1],
    c=df["cluster_kmeans"],
    cmap="tab10",
    s=50
)
plt.scatter(
    centroides[:,0], centroides[:,1],
    c="red", s=200, marker="X", label="Centroides"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusters K-Means sobre PCA")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_pca.png", dpi=300)
plt.close()

print("\n" + "="*80)
print("10. ANÁLISIS DE CORRELACIÓN")
print("="*80)

corr_matrix = df_scaled.corr()

plt.figure(figsize=(10,8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    square=True,
    linewidths=0.5
)
plt.title("Matriz de Correlación - Variables Numéricas")
plt.tight_layout()
plt.savefig("matriz_correlacion.png", dpi=300)
plt.close()

print("Gráfico guardado como 'matriz_correlacion.png'")

print("\n" + "="*80)
print("8.x CARGAS FACTORIALES (Contribución de variables al PCA)")
print("="*80)

# DataFrame de cargas factoriales
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=df_scaled.columns
)

print("\nCargas factoriales (primeras filas):")
print(loadings.head())

n_top = 10  # número de variables a mostrar

for i in range(min(3, pca.n_components_)):
    pc = f"PC{i+1}"
    print(f"\nVariables más influyentes en {pc}:")
    
    top_vars = loadings[pc].abs().sort_values(ascending=False).head(n_top)
    for var in top_vars.index:
        value = loadings.loc[var, pc]
        direction = "positiva" if value > 0 else "negativa"
        print(f"  • {var}: {value:.3f} ({direction})")

plt.figure(figsize=(8,6))

top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(10).index
sns.barplot(
    x=loadings.loc[top_pc1, "PC1"],
    y=top_pc1,
    color="steelblue"
)

plt.title("Cargas factoriales - PC1")
plt.xlabel("Carga")
plt.ylabel("Variable")
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig("cargas_pc1.png", dpi=300)
plt.close()

if pca.n_components_ >= 2:
    plt.figure(figsize=(8,6))
    
    top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(10).index
    sns.barplot(
        x=loadings.loc[top_pc2, "PC2"],
        y=top_pc2,
        color="darkorange"
    )

    plt.title("Cargas factoriales - PC2")
    plt.xlabel("Carga")
    plt.ylabel("Variable")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig("cargas_pc2.png", dpi=300)
    plt.close()
