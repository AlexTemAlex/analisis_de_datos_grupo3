import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os  # Importar módulo os para manejo de carpetas

DATASET = "amazon_actualizado3.csv"
# -------------------------
# 2. CARGA DE DATOS (OPTIMIZADA)
# -------------------------
DF = pd.read_csv(
    DATASET,
    parse_dates=["OrderDate"],
    dtype={
        "OrderID": "string",
        "CustomerID": "string",
        "CustomerName": "string",
        "ProductID": "string",
        "ProductName": "string",
        "Category": "category",
        "Brand": "category",
        "PaymentMethod": "category",
        "OrderStatus": "category",
        "City": "category",
        "State": "category",
        "Country": "category",
        "SellerID": "string"
    }
)

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
print(f"   Shape: {DF.shape}")
print(f"   Total Registros: {DF.shape[0]:,}")
print(f"   Total Variables: {DF.shape[1]}")

print("\n1.1 Primeras 5 filas:")
print(DF.head())

print(f"\n1.3 Tipos de Variables Originales")
print(f"{DF.info()}\n")

print(f"{DF.nunique()}\n")
# -------------------------------------------------------------------------
# 2. LIMPIEZA BÁSICA
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("2. Eliminar registros duplicados:")
print("="*80)
duplicates = DF.duplicated().sum()
print(f"   Filas Duplicadas: {duplicates}")
if duplicates > 0:
    DF = DF.drop_duplicates()
    print(f"   {duplicates} filas duplicadas eliminadas")

# Conversión de fecha
DF['OrderDate'] = pd.to_datetime(DF['OrderDate'], errors='coerce')


# -------------------------------------------------------------------------
# 3. ELIMINACIÓN DE VARIABLES IRRELEVANTES
# -------------------------------------------------------------------------
drop_cols = [
    'OrderID',
    'CustomerName',
    'ProductName',
    'Country'
]
print("\n" + "="*80)
print("3. Eliminar variables irrelevantes")
print("="*80)
print(f"3.1 Variables eliminadas: {drop_cols}")
df = DF.drop(columns=drop_cols, errors='ignore')
print("3.2 Variables restantes:", df.columns.tolist())
# -------------------------------------------------------------------------
# 4. TRATAMIENTO DE VALORES NULOS
# -------------------------------------------------------------------------
print("\n" + "="*80)
print(f"4. Revisión de Valores Nulos")
print("="*80)
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({
    'Cantidad Nulos': missing_data,
    'Porcentaje': missing_percentage
})
missing_df = missing_summary[missing_summary['Cantidad Nulos'] > 0]
if len(missing_df) > 0:
    print(missing_df)
else:
    print(f"   4.1 Valores nulos: {len(missing_df)}")

print("\n" + "="*80)
print(f"4. Ganacia Total")
print("="*80)

ganancia_total = DF["TotalAmount"].sum()
print(ganancia_total)

print("\n" + "="*80)
print(f"4. Graficas de Dispersión")
print("="*80)

import itertools
pairplot_cols = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost', 'TotalAmount']
# if set(pairplot_cols).issubset(df.columns):
#     data = df[pairplot_cols].dropna()

#     # Todas las combinaciones sin repetición ni diagonal
#     pairs = list(itertools.combinations(pairplot_cols, 2))

#     # Crear figura 3x5
#     fig, axes = plt.subplots(3, 5, figsize=(20, 12))
#     axes = axes.flatten()

#     for ax, (y, x) in zip(axes, pairs):
#         sns.scatterplot(
#             data=data,
#             x=x,
#             y=y,
#             ax=ax
#         )
#         ax.set_xlabel(x)
#         ax.set_ylabel(y)

#     # Eliminar ejes vacíos (por seguridad)
#     for i in range(len(pairs), len(axes)):
#         fig.delaxes(axes[i])

#     fig.suptitle(
#         'Diagramas de Dispersión Bivariados (3 filas × 5 columnas)',
#         fontsize=16
#     )

#     fig.tight_layout()
#     fig.savefig(
#         "Diagrama_dispersion_3x5_bivariado.png",
#         dpi=300,
#         bbox_inches="tight"
#     )
# else:
#     print('Faltan algunas de las columnas requeridas para el gráfico de pares.')

print("\n" + "="*80)
print(f"4. Graficas de Distribución")
print("="*80)

from matplotlib.ticker import FuncFormatter

def human_format(x, pos=None):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.1f}k'
    else:
        return f'{x:.0f}'


# Grafica de tiempo
import matplotlib.dates as mdates

# Serie mensual
serie_mensual = df.set_index('OrderDate')['TotalAmount'].resample('M').sum()

fig, ax = plt.subplots(figsize=(12, 5))

# Crear colores por año
years = serie_mensual.index.year
unique_years = sorted(years.unique())
colors_map = {year: color for year, color in zip(unique_years, plt.cm.tab10.colors)}

# Dibujar línea que conecta todos los puntos
ax.plot(serie_mensual.index, serie_mensual.values, color='gray', linewidth=1.5, alpha=0.6)

# Graficar cada punto con color según el año
for date, value in serie_mensual.items():
    ax.plot(date, value, marker='o', color=colors_map[date.year], markersize=6)

# Etiquetas encima de cada punto: inicial del mes
for date, value in serie_mensual.items():
    month_initial = date.strftime('%b')[0]  # primera letra del mes
    ax.text(date, value + serie_mensual.max()*0.01, month_initial,
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Configuración de eje X
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Año')
ax.set_ylabel('TotalAmount')
ax.set_title('TotalAmount mensual por año con iniciales de mes')
ax.yaxis.set_major_formatter(FuncFormatter(human_format))

# Grid solo horizontal
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# Grafica de distribuciones numericas
numeric_vars = [
    'Quantity',
    'UnitPrice',
    'Discount',
    'Tax',
    'ShippingCost',
    'TotalAmount'
]

for col in numeric_vars:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma
    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Distribución de {col}')

    # Boxplot con outliers rojos
    sns.boxplot(
        y=df[col],
        ax=axes[1],
        flierprops=dict(
            marker='o',
            markerfacecolor='red',
            markeredgecolor='red',
            markersize=5
        )
    )
    axes[1].set_title(f'Boxplot de {col}')

    plt.tight_layout()
    plt.show()


# Grafica de distribuciones categoricas
categorical_vars = [
    'Category',
    'Brand',
    'PaymentMethod',
    'OrderStatus',
    'City',
    'State'
]
for col in categorical_vars:

    freq = df[col].value_counts(sort=False)
    income = df.groupby(col)['TotalAmount'].sum().reindex(freq.index)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # -----------------------------
    # Distribución (frecuencia)
    # -----------------------------
    sns.barplot(
        x=freq.index,
        y=freq.values,
        ax=ax1,
        color='#6BAED6'  # azul visible
    )
    ax1.set_ylabel('Frecuencia')
    ax1.set_xlabel(col)

    # Grid solo para frecuencia
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # -----------------------------
    # Ingresos (línea)
    # -----------------------------
    ax2 = ax1.twinx()
    ax2.plot(
        freq.index,
        income.values,
        color='#E31A1C',
        marker='o',
        linewidth=2
    )
    ax2.yaxis.set_major_formatter(FuncFormatter(human_format))
    ax2.set_ylabel('Ingresos')

    if col == 'Brand':

        income_sorted = income.sort_values(ascending=False)
        top5 = income_sorted.head(5).index
        bottom5 = income_sorted.tail(5).index
        show_labels = set(top5).union(bottom5)

    else:
        show_labels = freq.index


    # Etiquetas de ingresos
    for x, y in zip(freq.index, income.values):
        if x in show_labels:
            ax2.annotate(
                human_format(y, None),
                (x, y),
                textcoords='offset points',
                xytext=(0, 6),
                ha='center',
                fontsize=9,
                color='black'
            )


    ax1.set_title(f'Distribución e Ingresos por {col}')

    ax1.tick_params(
        axis='x',
        rotation=90 if col in ['Brand', 'Category', 'City', 'State'] else 45
    )

    plt.tight_layout()
    plt.show()


id_vars = ['CustomerID', 'ProductID', 'SellerID']

for col in id_vars:

    # Frecuencia (distribución)
    freq = df[col].value_counts()
    order = freq.index

    # Ingresos
    income = (
        df.groupby(col)['TotalAmount']
          .sum()
          .reindex(order)
    )

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # -----------------------------
    # Barras: distribución
    # -----------------------------
    sns.countplot(
        data=df,
        x=col,
        order=order,
        ax=ax1,
        color='#9ecae1'
    )
    ax1.set_ylabel('Frecuencia')
    ax1.set_xlabel('')
    ax1.set_xticks([])
    ax1.set_title(f'Distribución e Ingresos por {col}')

    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # -----------------------------
    # Puntos rojos: ingresos - Top 20 y Bottom 20 ingresos
    # -----------------------------
    income_sorted = income.sort_values(ascending=False)
    top20 = income_sorted.head(20).index
    bottom20 = income_sorted.tail(20).index

    selected_ids = [i for i in order if i in top20.union(bottom20)]
    selected_pos = [order.get_loc(i) for i in selected_ids]
    selected_income = income.loc[selected_ids]
    
    ax2 = ax1.twinx()
    ax2.scatter(
        selected_pos,
        selected_income.values,
        color='red',
        alpha=0.6,
        s=25,
        zorder=5
    )

    ax2.yaxis.set_major_formatter(FuncFormatter(human_format))
    ax2.set_ylabel('Ingresos')

    plt.tight_layout()
    plt.show()

for col in id_vars:

    # Sumar ingresos por ID
    income = df.groupby(col)['TotalAmount'].sum().sort_values(ascending=False)
    top10 = income.head(10)
    bottom10 = income.tail(10)
    combined = pd.concat([top10, bottom10])

    # -----------------------------
    # Reemplazar IDs por nombres
    # -----------------------------
    labels = pd.Series(combined.index.astype(str))  # convertir a Series para map

    if col == 'CustomerID':
        name_map = DF[['CustomerID', 'CustomerName']].drop_duplicates().set_index('CustomerID')['CustomerName']
        labels = labels.map(lambda x: name_map.get(x, x))

    elif col == 'ProductID':
        name_map = DF[['ProductID', 'ProductName']].drop_duplicates().set_index('ProductID')['ProductName']
        labels = labels.map(lambda x: name_map.get(x, x))

    # SellerID no tiene nombre → se deja tal cual

    # -----------------------------
    # Gráfica
    # -----------------------------
    plt.figure(figsize=(14, 5))
    ax = sns.barplot(
        x=labels,
        y=combined.values,
        palette=['red'] * 10 + ['blue'] * 10
    )

    plt.title(f'Top 10 y Bottom 10 Ingresos por {col}')
    plt.xlabel('')
    ax.yaxis.set_major_formatter(FuncFormatter(human_format))
    plt.ylabel('Ingresos')
    plt.xticks(rotation=90)

    # -----------------------------
    # Etiquetas encima de las barras
    # -----------------------------
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{height:,.1f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            fontsize=9,
            xytext=(0, 3),
            textcoords='offset points'
        )

    plt.tight_layout()
    plt.show()




# Unir df con DF para obtener ProductName
df_full = df.merge(
    DF[['ProductID', 'ProductName']].drop_duplicates(),
    on='ProductID',
    how='left'
)


# =========================
# Top 1 y Bottom 1 por categoría
# =========================

# Lista para almacenar los datos de todas las categorías
plot_data = []

# Iterar por categorías
for cat in df_full['Category'].unique():
    df_cat = df_full[df_full['Category'] == cat]
    
    # Sumar ingresos por producto
    prod_income = df_cat.groupby('ProductName')['TotalAmount'].sum()
    
    # Top 1 y Bottom 1
    top1 = prod_income.sort_values(ascending=False).head(1)
    bottom1 = prod_income.sort_values(ascending=True).head(1)
    
    combined = pd.concat([top1, bottom1])
    
    # Guardar los datos en formato de lista de diccionarios
    for product, value in combined.items():
        plot_data.append({
            'Category': cat,
            'Product': product,
            'TotalAmount': value,
            'Type': 'Top 1' if product in top1.index else 'Bottom 1'
        })

# Convertir en DataFrame
df_plot = pd.DataFrame(plot_data)

# Crear etiquetas combinando categoría y producto
df_plot['Label'] = df_plot['Category'] + ' - ' + df_plot['Product']

# Colores
palette = {'Top 1': 'red', 'Bottom 1': 'blue'}

# Gráfico único
plt.figure(figsize=(16,6))
ax = sns.barplot(
    x='Label',
    y='TotalAmount',
    hue='Type',
    data=df_plot,
    palette=palette
)

# Formato eje y
ax.yaxis.set_major_formatter(FuncFormatter(human_format))
plt.xticks(rotation=90, ha='right')
plt.ylabel('Ingresos')
plt.xlabel('Producto - Categoría')
plt.title('Top 1 y Bottom 1 productos por categoría')

# Etiquetas encima de cada barra
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        human_format(height, None),
        (p.get_x() + p.get_width()/2, height),
        ha='center',
        va='bottom',
        fontsize=9,
        xytext=(0,3),
        textcoords='offset points'
    )

plt.tight_layout()
plt.show()


# =========================
# Top 1 y Bottom 1 por estado
# =========================

# Lista para guardar datos de todos los estados
plot_data = []

# Iteramos por cada estado
for state in df_full['State'].unique():
    df_state = df_full[df_full['State'] == state]
    
    # Sumar ingresos totales por producto
    prod_income = df_state.groupby('ProductName')['TotalAmount'].sum()
    
    # Top 1 y Bottom 1
    top1 = prod_income.sort_values(ascending=False).head(1)
    bottom1 = prod_income.sort_values(ascending=True).head(1)
    
    combined = pd.concat([top1, bottom1])
    
    # Guardar en lista
    for prod, income in combined.items():
        plot_data.append({
            'State': state,
            'Product': prod,
            'TotalAmount': income,
            'Type': 'Top 1' if prod in top1.index else 'Bottom 1'
        })

# Convertir a DataFrame
df_plot = pd.DataFrame(plot_data)

# Crear etiquetas combinando producto y estado
df_plot['Label'] = df_plot['Product'] + '\n(' + df_plot['State'] + ')'

# Colores
palette = {'Top 1': 'red', 'Bottom 1': 'blue'}

# Gráfico vertical
plt.figure(figsize=(18,6))
ax = sns.barplot(
    x='Label',
    y='TotalAmount',
    hue='Type',
    data=df_plot,
    palette=palette
)

plt.title('Top 1 y Bottom 1 productos por estado')
plt.xlabel('Producto (Estado)')
plt.ylabel('Ingresos')
ax.yaxis.set_major_formatter(FuncFormatter(human_format))

# Etiquetas encima de cada barra
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        human_format(height, None),
        (p.get_x() + p.get_width()/2, height),
        ha='center',
        va='bottom',
        fontsize=9,
        xytext=(0,3),
        textcoords='offset points'
    )

plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.show()



# resumen_productos_mas_vendidos = (
#     DF.groupby("ProductName")
#       .agg(
#           UnidadesVendidas=("Quantity", "sum"),
#           IngresoTotal=("TotalAmount", "sum")
#       )
#       .sort_values("UnidadesVendidas", ascending=False)
#       .head(5)
# )
# print(resumen_productos_mas_vendidos)

# top_productos_ingresos = (
#     DF.groupby("ProductName")
#       .agg(
#           UnidadesVendidas=("Quantity", "sum"),
#           IngresoTotal=("TotalAmount", "sum")
#       )
#       .sort_values("IngresoTotal", ascending=False)
#       .head(5)
# )
# print(top_productos_ingresos)

# top_clientes_ingresos = (
#     DF.groupby(["CustomerID", "CustomerName"])
#       .agg(
#           TotalCompras=("Quantity", "sum"),
#           IngresoTotal=("TotalAmount", "sum")
#       )
#       .sort_values("IngresoTotal", ascending=False)
#       .head(5)
# )
# print(top_clientes_ingresos)




# # -------------------------------------------------------------------------
# # 7. ESTADÍSTICAS BÁSICAS
# # -------------------------------------------------------------------------
# print("\n" + "="*80)
# print("\n7.1 ESTADÍSTICAS BÁSICAS - Variables Numéricas:")
# print("\n" + "="*80)
# print(df.describe())

# print("\n" + "="*80)
# print("\n7.2 ESTADÍSTICAS BÁSICAS - Variables Categóricas:")
# print("="*80)
# print(df.describe(include="object").T)

# --- CREAR CARPETA PARA INSIGHTS ---
# INSIGHTS_FOLDER = "insights"
# if not os.path.exists(INSIGHTS_FOLDER):
#     os.makedirs(INSIGHTS_FOLDER)
#     print(f"\n📁 Carpeta '{INSIGHTS_FOLDER}' creada para guardar gráficos de insights")


# =============================================================================
# 1. AGREGACIÓN POR CLIENTE
# =============================================================================
clientes_agg = df.groupby('CustomerID').agg({
    'TotalAmount': ['sum', 'mean', 'count'],
    'Quantity': 'sum',
    'Discount': 'mean',
    'UnitPrice': 'mean',
    'OrderDate': 'max'
}).round(2)

clientes_agg.columns = [
    'Ingreso_Total', 'Ticket_Promedio', 'Frecuencia',
    'Cantidad_Total', 'Descuento_Promedio',
    'Precio_Promedio', 'Ultima_Compra'
]

clientes_agg['Dias_Ultima_Compra'] = (
    pd.Timestamp.now() - clientes_agg['Ultima_Compra']
).dt.days

print("\n" + "="*80)
print("1. AGREGACIÓN DE CLIENTES")
print("="*80)
print(f"Clientes analizados: {len(clientes_agg)}")


# =============================================================================
# 2. VARIABLES PARA CLUSTERING
# =============================================================================
clustering_vars = [
    'Ingreso_Total',
    'Ticket_Promedio',
    'Frecuencia',
    'Descuento_Promedio',
    'Dias_Ultima_Compra'
]

# =============================================================================
# 3. ESTANDARIZACIÓN
# =============================================================================
scaler = StandardScaler()
clientes_scaled = pd.DataFrame(
    scaler.fit_transform(clientes_agg[clustering_vars].fillna(0)),
    columns=clustering_vars,
    index=clientes_agg.index
)

print("\n" + "="*80)
print("2. ESTANDARIZACIÓN")
print("="*80)
print(clientes_scaled.describe().loc[['mean', 'std']])
# =============================================================================
# 4. PCA (2 COMPONENTES)
# =============================================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(clientes_scaled)

explained_var = pca.explained_variance_ratio_

print("\n" + "="*80)
print("3. PCA")
print("="*80)
print(f"Varianza explicada PC1 + PC2: {explained_var.sum():.2%}")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=clustering_vars
)

print("\nLoadings PCA:")
print(loadings)


# =============================================================================
# 4.1 VISUALIZACIÓN PCA (ANTES DE CLUSTERING)
# =============================================================================
plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6, color='gray')

plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
plt.title("Proyección PCA de Clientes (Sin Clustering)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# 5. MÉTODO DEL CODO (INERCIA)
# =============================================================================
K_range = range(2, 11)
inertia = []

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=20,
        max_iter=100,
        algorithm='lloyd',
        random_state=42
    )
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del Codo (Clientes - PCA)")
plt.grid(True)
plt.tight_layout()
plt.show()


# =============================================================================
# 6. K-MEANS FINAL
# =============================================================================
k_opt = 4

kmeans = KMeans(
    n_clusters=k_opt,
    init='k-means++',
    n_init=50,
    max_iter=100,
    algorithm='lloyd',
    random_state=42
)

clientes_agg['Segmento'] = kmeans.fit_predict(X_pca)


# =============================================================================
# 7. ANÁLISIS DE INERCIAS
# =============================================================================
inertia_intra = kmeans.inertia_

global_center = X_pca.mean(axis=0)
inertia_total = np.sum((X_pca - global_center) ** 2)

inertia_inter = inertia_total - inertia_intra

print("\n" + "="*80)
print("4. ANÁLISIS DE INERCIAS")
print("="*80)
print(f"Inercia total:      {inertia_total:.4f}")
print(f"Inercia intraclase: {inertia_intra:.4f} ({100*inertia_intra/inertia_total:.2f}%)")
print(f"Inercia interclase: {inertia_inter:.4f} ({100*inertia_inter/inertia_total:.2f}%)")

# =============================================================================
# 8. PCA + CLUSTERS + CÍRCULO DE CORRELACIONES (ETIQUETAS AUTOMÁTICAS + RADIO = 10)
# =============================================================================

# ----------------------------
# 8.1 PERFIL DE CLUSTERS
# ----------------------------
perfil_segmentos = clientes_agg.groupby('Segmento')[clustering_vars].mean()
# Relativo al promedio global
perfil_relativo = perfil_segmentos / clientes_agg[clustering_vars].mean()
# ----------------------------
# ASIGNACIÓN FLEXIBLE DE ETIQUETAS
# ----------------------------
segmento_labels = {}
for seg, row in perfil_relativo.iterrows():

    # 1. Premium: alto ingreso + alta frecuencia + reciente
    if row['Ingreso_Total'] > 1.2 and row['Frecuencia'] > 1.2 and row['Dias_Ultima_Compra'] < 0.8:
        segmento_labels[seg] = '🎯 PREMIUM (Alto Valor)'

    # 2. Frecuentes: alta frecuencia (aunque ingreso medio)
    elif row['Frecuencia'] > 1.1:
        segmento_labels[seg] = '🔄 FRECUENTES (Leales)'

    # 3. Ocasionales / Sensibles al precio: alta sensibilidad a descuento
    elif row['Descuento_Promedio'] > 1.05:
        segmento_labels[seg] = '💰 OCASIONALES (Sensibles Precio)'

    # 4. Inactivos: recencia alta + frecuencia baja
    else:
        segmento_labels[seg] = '⏰ INACTIVOS (Riesgo Pérdida)'

print("\nEtiquetas asignadas automáticamente (flexible):")
for k, v in segmento_labels.items():
    print(f"Cluster {k}: {v}")


# ----------------------------
# 8.3 PLOTEO FINAL
# ----------------------------
centroids = kmeans.cluster_centers_
plt.figure(figsize=(9,8))

# Clusters con etiquetas
for c in range(k_opt):
    plt.scatter(
        X_pca[clientes_agg['Segmento'] == c, 0],
        X_pca[clientes_agg['Segmento'] == c, 1],
        label=segmento_labels[c],
        alpha=0.6
    )

# Centroides
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X',
    s=200,
    color='black',
    label='Centroides'
)

# Círculo de correlaciones (radio = 10)
circle_radius = 10
theta = np.linspace(0, 2*np.pi, 300)
plt.plot(
    circle_radius * np.cos(theta),
    circle_radius * np.sin(theta),
    linestyle='--',
    color='gray',
    alpha=0.5
)

# Vectores + etiquetas de variables
for var in loadings.index:
    x = loadings.loc[var, 'PC1'] * circle_radius
    y = loadings.loc[var, 'PC2'] * circle_radius

    plt.plot([0, x], [0, y], color='black', linewidth=2)

    plt.text(
        x * 1.05,
        y * 1.05,
        var,
        fontsize=10,
        color='black',
        ha='center',
        va='center'
    )

# Ejes
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
plt.title("Segmentación de Clientes (PCA + K-Means + Círculo de Correlaciones)")
plt.legend()
plt.grid(alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()


# ============================
# Gráfico de pastel: Distribución de registros por cluster
# ============================

# Contar registros en df_full para cada cluster
record_counts = []

for c in range(k_opt):
    cluster_customers = clientes_agg[clientes_agg['Segmento'] == c].index
    n_records = df_full[df_full['CustomerID'].isin(cluster_customers)].shape[0]
    record_counts.append(n_records)

record_counts = np.array(record_counts)
record_labels = [f"Cluster {i}\n{segmento_labels[i]}" for i in range(k_opt)]

# Graficar
plt.figure(figsize=(8,8))
plt.pie(
    record_counts,
    labels=record_labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.tab10.colors[:k_opt],
    wedgeprops={'edgecolor': 'black'}
)
plt.title("Distribución de registros por segmento (K-Means)")
plt.show()



# ============================
# 1. Gráfico de pastel: Distribución de segmentos
# ============================

segment_counts = clientes_agg['Segmento'].value_counts().sort_index()
segment_labels = [f"Cluster {i}\n{segmento_labels[i]}" for i in segment_counts.index]

plt.figure(figsize=(8,8))
plt.pie(
    segment_counts,
    labels=segment_labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.tab10.colors[:len(segment_counts)],
    wedgeprops={'edgecolor': 'black'}
)
plt.title("Distribución de clientes por segmentos (K-Means)")
plt.show()


# ============================
# 2. Gráfico de radar (estrellas) por cluster
# ============================

clustering_vars = [
    'Ingreso_Total',
    'Ticket_Promedio',
    'Frecuencia',
    'Descuento_Promedio',
    'Dias_Ultima_Compra'
]

# Promedios por cluster
cluster_means = clientes_agg.groupby('Segmento')[clustering_vars].mean()

# Normalizar cada variable para que estén entre 0 y 1
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

# Preparar ángulos
num_vars = len(clustering_vars)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # cerrar el círculo

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

# Dibujar círculos concéntricos
ax.set_rscale('linear')
ax.set_ylim(0, 1.1)

# Variables como etiquetas
plt.xticks(angles[:-1], clustering_vars, fontsize=10)

# Graficar cada cluster
colors = plt.cm.tab10.colors
for i, row in cluster_means_norm.iterrows():
    values = row.tolist()
    values += values[:1]  # cerrar el círculo
    ax.plot(angles, values, color=colors[i], linewidth=2, label=f"Cluster {i} ({segmento_labels[i]})")
    ax.fill(angles, values, color=colors[i], alpha=0.25)  # relleno semitransparente

# Círculos de referencia
ax.yaxis.grid(True, color='gray', linestyle='--', alpha=0.5)
ax.xaxis.grid(True, color='gray', linestyle='--', alpha=0.5)

plt.title("Perfil de clusters (Radar / Estrella)", size=14, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# ==============================
# Perfil detallado por cluster (extendido)
# ==============================

cluster_stats = []

for c in range(k_opt):
    # Tomamos los clientes de este cluster
    cluster_customers = clientes_agg[clientes_agg['Segmento'] == c].index
    
    # Filtrar registros de df_full usando CustomerID
    df_cluster = df_full[df_full['CustomerID'].isin(cluster_customers)]
    
    n_records = len(df_cluster)
    pct_total = n_records / len(df_full) * 100
    total_ingresos = df_cluster['TotalAmount'].sum()
    
    # Rango de precios
    min_price = df_cluster['UnitPrice'].min()
    max_price = df_cluster['UnitPrice'].max()
    
    # Top 3 y Bottom 3 productos por ingreso
    prod_income = df_cluster.groupby('ProductName')['TotalAmount'].sum()
    top3_prod = prod_income.sort_values(ascending=False).head(3).to_dict()
    bottom3_prod = prod_income.sort_values(ascending=True).head(3).to_dict()
    
    # Top 3 y Bottom 3 categorías por ingreso
    cat_income = df_cluster.groupby('Category')['TotalAmount'].sum()
    top3_cat = cat_income.sort_values(ascending=False).head(3).to_dict()
    bottom3_cat = cat_income.sort_values(ascending=True).head(3).to_dict()
    
    # Top 3 y Bottom 3 productos por cantidad
    prod_qty = df_cluster.groupby('ProductName')['Quantity'].sum()
    top3_qty_prod = prod_qty.sort_values(ascending=False).head(3).to_dict()
    bottom3_qty_prod = prod_qty.sort_values(ascending=True).head(3).to_dict()
    
    # Top 3 y Bottom 3 categorías por cantidad
    cat_qty = df_cluster.groupby('Category')['Quantity'].sum()
    top3_qty_cat = cat_qty.sort_values(ascending=False).head(3).to_dict()
    bottom3_qty_cat = cat_qty.sort_values(ascending=True).head(3).to_dict()
    
    # Estados presentes
    estados = df_cluster['State'].value_counts().to_dict()
    
    # Estadísticas por usuario
    user_stats = df_cluster.groupby('CustomerID').agg({'Quantity':'sum', 'TotalAmount':'sum'})
    
    # Usuario con más compras (cantidad)
    top_user_qty = user_stats['Quantity'].idxmax()
    top_user_qty_value = user_stats.loc[top_user_qty, 'Quantity']
    top_user_qty_income = user_stats.loc[top_user_qty, 'TotalAmount']
    
    # Usuario con menos compras (cantidad)
    bottom_user_qty = user_stats['Quantity'].idxmin()
    bottom_user_qty_value = user_stats.loc[bottom_user_qty, 'Quantity']
    bottom_user_qty_income = user_stats.loc[bottom_user_qty, 'TotalAmount']
    
    # Usuario con más ingreso
    top_user_income = user_stats['TotalAmount'].idxmax()
    top_user_income_value = user_stats.loc[top_user_income, 'TotalAmount']
    top_user_income_qty = user_stats.loc[top_user_income, 'Quantity']
    
    # Usuario con menos ingreso
    bottom_user_income = user_stats['TotalAmount'].idxmin()
    bottom_user_income_value = user_stats.loc[bottom_user_income, 'TotalAmount']
    bottom_user_income_qty = user_stats.loc[bottom_user_income, 'Quantity']
    
    # Cantidad de clientes únicos
    n_customers = df_cluster['CustomerID'].nunique()
    
    # Cantidad de productos únicos
    n_products = df_cluster['ProductID'].nunique() if 'ProductID' in df_cluster.columns else df_cluster['ProductName'].nunique()
    
    # Cantidad de vendedores únicos
    n_sellers = df_cluster['SellerID'].nunique() if 'SellerID' in df_cluster.columns else None
    
    # Guardar todo en diccionario
    cluster_stats.append({
        'Cluster': c,
        'Etiqueta': segmento_labels[c],
        'Registros': n_records,
        'Porcentaje_total': pct_total,
        'Clientes_unicos': n_customers,
        'Productos_unicos': n_products,
        'Vendedores_unicos': n_sellers,
        'Ingresos_totales': total_ingresos,
        'Precio_min': min_price,
        'Precio_max': max_price,
        'Top3_Productos_Ingreso': top3_prod,
        'Bottom3_Productos_Ingreso': bottom3_prod,
        'Top3_Categorias_Ingreso': top3_cat,
        'Bottom3_Categorias_Ingreso': bottom3_cat,
        'Top3_Productos_Cantidad': top3_qty_prod,
        'Bottom3_Productos_Cantidad': bottom3_qty_prod,
        'Top3_Categorias_Cantidad': top3_qty_cat,
        'Bottom3_Categorias_Cantidad': bottom3_qty_cat,
        'Estados': estados,
        'Usuario_Mas_Compras': {'CustomerID': top_user_qty, 'Cantidad': top_user_qty_value, 'Ingresos': top_user_qty_income},
        'Usuario_Menos_Compras': {'CustomerID': bottom_user_qty, 'Cantidad': bottom_user_qty_value, 'Ingresos': bottom_user_qty_income},
        'Usuario_Mas_Ingreso': {'CustomerID': top_user_income, 'Ingresos': top_user_income_value, 'Cantidad': top_user_income_qty},
        'Usuario_Menos_Ingreso': {'CustomerID': bottom_user_income, 'Ingresos': bottom_user_income_value, 'Cantidad': bottom_user_income_qty},
    })

# Convertir a DataFrame para mejor visualización
df_clusters = pd.DataFrame(cluster_stats)

# Mostrar el resumen completo
pd.set_option('display.max_colwidth', None)
print(df_clusters)



# Función para formatear números grandes
def human_format(num, pos=None):
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000:
            return f"{num:.0f}{unit}"
        num /= 1000
    return f"{num:.0f}B"

# -----------------------------
# Generar gráficos por cluster
# -----------------------------
for c in range(k_opt):
    df_cluster = df_full[df_full['CustomerID'].isin(clientes_agg[clientes_agg['Segmento']==c].index)]
    etiqueta = segmento_labels[c]
    
    # --- Productos por ingreso ---
    prod_income = df_cluster.groupby('ProductName')['TotalAmount'].sum()
    top3 = prod_income.sort_values(ascending=False).head(3)
    bottom3 = prod_income.sort_values(ascending=True).head(3)
    combined = pd.concat([top3, bottom3])
    
    labels = combined.index
    colors = ['red']*3 + ['blue']*3
    
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=labels, y=combined.values, palette=colors)
    plt.title(f'Cluster {c} ({etiqueta}) - Top/Bottom 3 Productos por Ingreso')
    plt.ylabel('Total Amount')
    plt.xlabel('Producto')
    
    # Etiquetas encima de las barras
    for p in ax.patches:
        ax.annotate(human_format(p.get_height()),
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # --- Productos por cantidad ---
    prod_qty = df_cluster.groupby('ProductName')['Quantity'].sum()
    top3_qty = prod_qty.sort_values(ascending=False).head(3)
    bottom3_qty = prod_qty.sort_values(ascending=True).head(3)
    combined_qty = pd.concat([top3_qty, bottom3_qty])
    
    labels = combined_qty.index
    colors = ['red']*3 + ['blue']*3
    
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=labels, y=combined_qty.values, palette=colors)
    plt.title(f'Cluster {c} ({etiqueta}) - Top/Bottom 3 Productos por Cantidad')
    plt.ylabel('Cantidad')
    plt.xlabel('Producto')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # --- Categorías por ingreso ---
    cat_income = df_cluster.groupby('Category')['TotalAmount'].sum()
    top3_cat = cat_income.sort_values(ascending=False).head(3)
    bottom3_cat = cat_income.sort_values(ascending=True).head(3)
    combined_cat = pd.concat([top3_cat, bottom3_cat])
    
    labels = combined_cat.index
    colors = ['red']*3 + ['blue']*3
    
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=labels, y=combined_cat.values, palette=colors)
    plt.title(f'Cluster {c} ({etiqueta}) - Top/Bottom 3 Categorías por Ingreso')
    plt.ylabel('Total Amount')
    plt.xlabel('Categoría')
    
    for p in ax.patches:
        ax.annotate(human_format(p.get_height()),
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # --- Categorías por cantidad ---
    cat_qty = df_cluster.groupby('Category')['Quantity'].sum()
    top3_cat_qty = cat_qty.sort_values(ascending=False).head(3)
    bottom3_cat_qty = cat_qty.sort_values(ascending=True).head(3)
    combined_cat_qty = pd.concat([top3_cat_qty, bottom3_cat_qty])
    
    labels = combined_cat_qty.index
    colors = ['red']*3 + ['blue']*3
    
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=labels, y=combined_cat_qty.values, palette=colors)
    plt.title(f'Cluster {c} ({etiqueta}) - Top/Bottom 3 Categorías por Cantidad')
    plt.ylabel('Cantidad')
    plt.xlabel('Categoría')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# # ============================================================================
# # NUEVO PUNTO 12: ANÁLISIS DE PRODUCTOS MÁS RENTABLES (INSIGHT 1)
# # ============================================================================

# def analisis_productos_rentables(df):
#     """
#     Insight 1: Análisis de productos más rentables (Regla 80/20)
#     """
#     print("\n" + "="*80)
#     print("12. ANÁLISIS DE PRODUCTOS MÁS RENTABLES - REGLA 80/20")
#     print("="*80)
    
#     # 1. Productos por ingresos totales
#     print("\n1. TOP 10 PRODUCTOS POR INGRESOS TOTALES:")
#     productos_ingresos = df.groupby('ProductName')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
#     productos_ingresos = productos_ingresos.rename(columns={
#         'sum': 'Ingreso_Total',
#         'count': 'Cantidad_Ventas',
#         'mean': 'Ticket_Promedio'
#     }).sort_values('Ingreso_Total', ascending=False)
    
#     print(productos_ingresos.head(10))
    
#     # 2. Análisis 80/20 (Pareto)
#     print("\n2. ANÁLISIS PARETO (80/20):")
#     productos_ingresos_sorted = productos_ingresos.sort_values('Ingreso_Total', ascending=False)
#     productos_ingresos_sorted['Ingreso_Acumulado'] = productos_ingresos_sorted['Ingreso_Total'].cumsum()
#     productos_ingresos_sorted['%_Acumulado'] = (productos_ingresos_sorted['Ingreso_Acumulado'] / 
#                                                 productos_ingresos_sorted['Ingreso_Total'].sum() * 100)
    
#     # Encontrar qué productos generan el 80% de ingresos
#     productos_80 = productos_ingresos_sorted[productos_ingresos_sorted['%_Acumulado'] <= 80]
#     n_productos_80 = len(productos_80)
#     total_productos = len(productos_ingresos_sorted)
#     porcentaje_productos = (n_productos_80 / total_productos) * 100
    
#     print(f"   • Total productos: {total_productos}")
#     print(f"   • Productos que generan 80% de ingresos: {n_productos_80}")
#     print(f"   • Esto representa el {porcentaje_productos:.1f}% de todos los productos")
#     print(f"   • {100 - porcentaje_productos:.1f}% de productos generan solo 20% de ingresos")
    
#     # Mostrar los productos clave
#     print(f"\n   PRODUCTOS CLAVE (generan 80% de ingresos):")
#     for i, (producto, fila) in enumerate(productos_80.head(15).iterrows(), 1):
#         print(f"   {i:2d}. {producto[:40]:40s} | ${fila['Ingreso_Total']:>10,.0f} | {fila['Cantidad_Ventas']:>4} ventas")
    
#     # 3. Categorías más rentables
#     print("\n3. CATEGORÍAS MÁS RENTABLES:")
#     categorias_ingresos = df.groupby('Category')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
#     categorias_ingresos = categorias_ingresos.rename(columns={
#         'sum': 'Ingreso_Total',
#         'count': 'Cantidad_Ventas',
#         'mean': 'Ticket_Promedio'
#     }).sort_values('Ingreso_Total', ascending=False)
    
#     print(categorias_ingresos)
    
#     # 4. Marcas más rentables
#     print("\n4. MARCAS MÁS RENTABLES:")
#     marcas_ingresos = df.groupby('Brand')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
#     marcas_ingresos = marcas_ingresos.rename(columns={
#         'sum': 'Ingreso_Total',
#         'count': 'Cantidad_Ventas',
#         'mean': 'Ticket_Promedio'
#     }).sort_values('Ingreso_Total', ascending=False)
    
#     print(marcas_ingresos.head(10))
    
#     # 5. Visualizaciones
#     print("\n5. VISUALIZACIONES GUARDADAS:")
    
#     # Gráfico 1: Top 10 productos por ingresos
#     plt.figure(figsize=(12, 6))
#     top_10_productos = productos_ingresos.head(10)
#     bars = plt.barh(range(len(top_10_productos)), top_10_productos['Ingreso_Total'], 
#                    color='skyblue', edgecolor='black')
#     plt.yticks(range(len(top_10_productos)), top_10_productos.index, fontsize=9)
#     plt.xlabel('Ingreso Total ($)', fontsize=12)
#     plt.title('Top 10 Productos por Ingresos Totales', fontsize=14, fontweight='bold')
    
#     # Añadir valores en las barras
#     for i, bar in enumerate(bars):
#         width = bar.get_width()
#         plt.text(width + max(top_10_productos['Ingreso_Total']) * 0.01, 
#                 bar.get_y() + bar.get_height()/2,
#                 f'${width:,.0f}', ha='left', va='center', fontsize=9)
    
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
#     plt.savefig(f'{INSIGHTS_FOLDER}/top_10_productos.png', dpi=300, bbox_inches='tight')
#     print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/top_10_productos.png'")
    
#     # Gráfico 2: Análisis Pareto
#     fig, ax1 = plt.subplots(figsize=(10, 6))
    
#     # Barras de ingresos
#     ax1.bar(range(len(productos_ingresos_sorted.head(20))), 
#            productos_ingresos_sorted.head(20)['Ingreso_Total'],
#            color='lightblue', alpha=0.7, label='Ingreso por Producto')
#     ax1.set_xlabel('Productos (ordenados por ingresos)')
#     ax1.set_ylabel('Ingreso Total ($)', color='navy')
#     ax1.tick_params(axis='y', labelcolor='navy')
    
#     # Línea de Pareto
#     ax2 = ax1.twinx()
#     ax2.plot(range(len(productos_ingresos_sorted.head(20))),
#             productos_ingresos_sorted.head(20)['%_Acumulado'],
#             color='red', marker='o', linewidth=2, markersize=4,
#             label='% Acumulado')
#     ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80%')
#     ax2.set_ylabel('% Ingreso Acumulado', color='red')
#     ax2.tick_params(axis='y', labelcolor='red')
#     ax2.set_ylim(0, 100)
    
#     plt.title('Análisis Pareto - Ingresos por Producto', fontsize=14, fontweight='bold')
#     plt.xticks(range(len(productos_ingresos_sorted.head(20))), 
#               [f'P{i+1}' for i in range(20)], rotation=45)
    
#     # Combinar leyendas
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig(f'{INSIGHTS_FOLDER}/analisis_pareto.png', dpi=300, bbox_inches='tight')
#     print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/analisis_pareto.png'")
#     plt.close('all')  # Cerrar todas las figuras para liberar memoria
    
#     # 6. Recomendaciones específicas
#     print("\n6. RECOMENDACIONES DE NEGOCIO:")
#     print("   ✓ ENFOCAR INVENTARIO en los productos del top 20% (ver archivo 'analisis_pareto.png')")
#     print("   ✓ CREAR BUNDLES con productos complementarios de alta rentabilidad")
#     print("   ✓ NEGOCIAR MEJORES CONDICIONES con marcas del top 10 (ver análisis de marcas)")
#     print("   ✓ OPTIMIZAR PRECIOS en categorías con mayor ticket promedio")
#     print("   ✓ DESARROLLAR CAMPOS DE VENTAS cruzadas entre productos del mismo cliente")
    
#     return productos_ingresos, productos_80

# # ============================================================================
# # NUEVO PUNTO 13: SEGMENTACIÓN DE CLIENTES (INSIGHT 3)
# # ============================================================================

# def segmentacion_clientes(df, n_clusters=4):
#     """
#     Insight 3: Segmentación de clientes en 4 grupos basados en comportamiento
#     """
#     print("\n" + "="*80)
#     print("13. SEGMENTACIÓN DE CLIENTES - 4 GRUPOS DE COMPORTAMIENTO")
#     print("="*80)
    
#     # 1. Preparar datos de clientes
#     print("\n1. PREPARACIÓN DE DATOS DE CLIENTES:")
    
#     # Calcular métricas por cliente
#     clientes_agg = df.groupby('CustomerID').agg({
#         'TotalAmount': ['sum', 'mean', 'count'],  # Ingreso total, ticket promedio, frecuencia
#         'Quantity': 'sum',                        # Cantidad total comprada
#         'Discount': 'mean',                       # Sensibilidad a descuentos
#         'UnitPrice': 'mean',                      # Precio promedio pagado
#         'OrderDate': 'max'                        # Última compra
#     }).round(2)
    
#     # Renombrar columnas
#     clientes_agg.columns = ['Ingreso_Total', 'Ticket_Promedio', 'Frecuencia',
#                            'Cantidad_Total', 'Descuento_Promedio', 
#                            'Precio_Promedio', 'Ultima_Compra']
    
#     # Calcular días desde última compra
#     clientes_agg['Dias_Ultima_Compra'] = (pd.Timestamp.now() - clientes_agg['Ultima_Compra']).dt.days
    
#     print(f"   • Total clientes analizados: {len(clientes_agg)}")
#     print(f"   • Métricas calculadas: Ingreso total, Ticket promedio, Frecuencia, etc.")
    
#     # 2. Estandarizar datos para clustering
#     print("\n2. APLICACIÓN DE CLUSTERING (K-Means):")
    
#     # Seleccionar variables para clustering
#     clustering_vars = ['Ingreso_Total', 'Ticket_Promedio', 'Frecuencia', 
#                       'Descuento_Promedio', 'Dias_Ultima_Compra']
    
#     # Estandarizar
#     scaler = StandardScaler()
#     clientes_scaled = pd.DataFrame(
#         scaler.fit_transform(clientes_agg[clustering_vars].fillna(0)),
#         columns=clustering_vars,
#         index=clientes_agg.index
#     )
    
#     # Aplicar K-Means
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     clientes_agg['Segmento'] = kmeans.fit_predict(clientes_scaled)
    
#     # 3. Analizar segmentos
#     print(f"\n3. CARACTERÍSTICAS DE LOS {n_clusters} SEGMENTOS:")
    
#     segmentos_analisis = clientes_agg.groupby('Segmento')[clustering_vars].agg(['mean', 'count']).round(2)
    
#     # Renombrar segmentos según características
#     segmento_nombres = {
#         0: '🎯 PREMIUM (Alto Valor)',
#         1: '🔄 FRECUENTES (Leales)',
#         2: '💰 OCASIONALES (Sensibles Precio)',
#         3: '⏰ INACTIVOS (Riesgo Pérdida)'
#     }
    
#     clientes_agg['Segmento_Nombre'] = clientes_agg['Segmento'].map(segmento_nombres)
    
#     print("\n   RESUMEN POR SEGMENTO:")
#     for seg_num, seg_nombre in segmento_nombres.items():
#         seg_data = clientes_agg[clientes_agg['Segmento'] == seg_num]
#         print(f"\n   {seg_nombre}:")
#         print(f"     • Número de clientes: {len(seg_data)}")
#         print(f"     • Ingreso total promedio: ${seg_data['Ingreso_Total'].mean():,.0f}")
#         print(f"     • Ticket promedio: ${seg_data['Ticket_Promedio'].mean():,.0f}")
#         print(f"     • Frecuencia promedio: {seg_data['Frecuencia'].mean():.1f} compras")
#         print(f"     • Días desde última compra: {seg_data['Dias_Ultima_Compra'].mean():.0f} días")
    
#     # 4. Top clientes por segmento
#     print("\n4. TOP 5 CLIENTES POR SEGMENTO:")
    
#     for seg_num, seg_nombre in segmento_nombres.items():
#         seg_clientes = clientes_agg[clientes_agg['Segmento'] == seg_num]
#         top_5 = seg_clientes.nlargest(5, 'Ingreso_Total')
        
#         print(f"\n   {seg_nombre}:")
#         for idx, (cliente_id, fila) in enumerate(top_5.iterrows(), 1):
#             print(f"     {idx}. Cliente {cliente_id}:")
#             print(f"        - Ingreso total: ${fila['Ingreso_Total']:,.0f}")
#             print(f"        - Ticket promedio: ${fila['Ticket_Promedio']:,.0f}")
#             print(f"        - Frecuencia: {fila['Frecuencia']:.0f} compras")
    
#     # 5. Visualizaciones
#     print("\n5. VISUALIZACIONES GUARDADAS:")
    
#     # Gráfico 1: Distribución de segmentos
#     plt.figure(figsize=(10, 6))
#     segment_counts = clientes_agg['Segmento_Nombre'].value_counts()
#     colors = ['gold', 'lightgreen', 'lightcoral', 'lightblue']
    
#     plt.pie(segment_counts.values, labels=segment_counts.index,
#            autopct='%1.1f%%', startangle=90, colors=colors,
#            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
#     plt.title('Distribución de Clientes por Segmento', fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(f'{INSIGHTS_FOLDER}/distribucion_segmentos.png', dpi=300, bbox_inches='tight')
#     print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/distribucion_segmentos.png'")
    
#     # Gráfico 2: Características por segmento (radar chart)
#     fig = plt.figure(figsize=(10, 8))
    
#     # Preparar datos para radar chart
#     segment_means = clientes_agg.groupby('Segmento_Nombre')[clustering_vars].mean()
    
#     # Normalizar para radar chart
#     segment_normalized = segment_means.copy()
#     for col in clustering_vars:
#         segment_normalized[col] = (segment_means[col] - segment_means[col].min()) / \
#                                  (segment_means[col].max() - segment_means[col].min())
    
#     # Crear radar chart
#     angles = np.linspace(0, 2 * np.pi, len(clustering_vars), endpoint=False).tolist()
#     angles += angles[:1]  # Cerrar el círculo
    
#     ax = fig.add_subplot(111, polar=True)
    
#     for idx, (seg_name, seg_data) in enumerate(segment_normalized.iterrows()):
#         values = seg_data.tolist()
#         values += values[:1]  # Cerrar el círculo
        
#         ax.plot(angles, values, 'o-', linewidth=2, label=seg_name)
#         ax.fill(angles, values, alpha=0.25)
    
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(clustering_vars, fontsize=10)
#     ax.set_ylim(0, 1)
#     ax.set_title('Características por Segmento de Clientes', fontsize=14, fontweight='bold', pad=20)
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
#     plt.tight_layout()
#     plt.savefig(f'{INSIGHTS_FOLDER}/radar_segmentos.png', dpi=300, bbox_inches='tight')
#     print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/radar_segmentos.png'")
#     plt.close('all')  # Cerrar todas las figuras para liberar memoria
    
#     # 6. Recomendaciones por segmento
#     print("\n6. ESTRATEGIAS POR SEGMENTO:")
    
#     estrategias = {
#         '🎯 PREMIUM (Alto Valor)': [
#             "✓ Programa de fidelización premium",
#             "✓ Atención personalizada (asesor dedicado)",
#             "✓ Acceso anticipado a nuevos productos",
#             "✓ Eventos exclusivos para el segmento"
#         ],
#         '🔄 FRECUENTES (Leales)': [
#             "✓ Programa de puntos por compras",
#             "✓ Descuentos por volumen/repetición",
#             "✓ Recomendaciones personalizadas",
#             "✓ Encuestas de satisfacción periódicas"
#         ],
#         '💰 OCASIONALES (Sensibles Precio)': [
#             "✓ Ofertas y promociones específicas",
#             "✓ Recordatorios de carrito abandonado",
#             "✓ Comparativas de precio vs competencia",
#             "✓ Programas de referidos con incentivos"
#         ],
#         '⏰ INACTIVOS (Riesgo Pérdida)': [
#             "✓ Campañas de reactivación (email/SMS)",
#             "✓ Ofertas de re-enganche",
#             "✓ Encuestas para entender causas",
#             "✓ Programa de win-back específico"
#         ]
#     }
    
#     for segmento, acciones in estrategias.items():
#         print(f"\n   {segmento}:")
#         for accion in acciones:
#             print(f"   {accion}")
    
#     return clientes_agg

# # ============================================================================
# # NUEVO PUNTO 12: ANÁLISIS TEMPORAL/ESTACIONALIDAD (INSIGHT 4)
# # ============================================================================

# def analisis_estacionalidad(df):
#     """
#     Insight 4: Análisis temporal y estacionalidad
#     """
#     print("\n" + "="*80)
#     print("14. ANÁLISIS TEMPORAL Y ESTACIONALIDAD")
#     print("="*80)
    
#     # 1. Ventas por año
#     print("\n1. VENTAS POR AÑO:")
#     ventas_anual = df.groupby('OrderYear')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
#     ventas_anual = ventas_anual.rename(columns={
#         'sum': 'Ingreso_Total',
#         'count': 'Cantidad_Pedidos',
#         'mean': 'Ticket_Promedio'
#     })
    
#     print(ventas_anual)
    
#     # Cálculo de crecimiento anual
#     if len(ventas_anual) > 1:
#         print("\n   CRECIMIENTO ANUAL:")
#         ventas_anual['Crecimiento_%'] = ventas_anual['Ingreso_Total'].pct_change() * 100
#         print(ventas_anual[['Ingreso_Total', 'Crecimiento_%']].round(2))
    
#     # 2. Ventas por mes (promedio)
#     print("\n2. VENTAS POR MES (PROMEDIO):")
    
#     # Mapeo de números de mes a nombres
#     meses_nombres = {
#         1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
#         5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
#         9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
#     }
    
#     ventas_mensual = df.groupby('OrderMonth')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
#     ventas_mensual = ventas_mensual.rename(columns={
#         'sum': 'Ingreso_Total',
#         'count': 'Cantidad_Pedidos',
#         'mean': 'Ticket_Promedio'
#     })
#     ventas_mensual.index = ventas_mensual.index.map(meses_nombres)
    
#     print(ventas_mensual)
    
#     # 3. Ventas por trimestre
#     print("\n3. VENTAS POR TRIMESTRE:")
#     trimestre_nombres = {1: 'Q1 (Ene-Mar)', 2: 'Q2 (Abr-Jun)', 
#                         3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dic)'}
    
#     ventas_trimestral = df.groupby('OrderQuarter')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
#     ventas_trimestral = ventas_trimestral.rename(columns={
#         'sum': 'Ingreso_Total',
#         'count': 'Cantidad_Pedidos',
#         'mean': 'Ticket_Promedio'
#     })
#     ventas_trimestral.index = ventas_trimestral.index.map(trimestre_nombres)
    
#     print(ventas_trimestral)
    
#     # 4. Días de la semana con más ventas
#     print("\n4. VENTAS POR DÍA DE LA SEMANA:")
#     df['Dia_Semana'] = df['OrderDate'].dt.day_name()
    
#     # Ordenar días de la semana
#     dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
#                  'Friday', 'Saturday', 'Sunday']
#     dias_espanol = {
#         'Monday': 'Lunes',
#         'Tuesday': 'Martes',
#         'Wednesday': 'Miércoles',
#         'Thursday': 'Jueves',
#         'Friday': 'Viernes',
#         'Saturday': 'Sábado',
#         'Sunday': 'Domingo'
#     }
    
#     ventas_diarias = df.groupby('Dia_Semana')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
#     ventas_diarias = ventas_diarias.rename(columns={
#         'sum': 'Ingreso_Total',
#         'count': 'Cantidad_Pedidos',
#         'mean': 'Ticket_Promedio'
#     })
    
#     # Reindexar para orden correcto
#     ventas_diarias = ventas_diarias.reindex(dias_orden)
#     ventas_diarias.index = ventas_diarias.index.map(dias_espanol)
    
#     print(ventas_diarias)
    
#     # 5. Identificar temporadas pico
#     print("\n5. IDENTIFICACIÓN DE TEMPORADAS PICO:")
    
#     # Meses con mayores ventas (top 3)
#     top_meses = ventas_mensual.nlargest(3, 'Ingreso_Total')
#     print("   MESES CON MAYORES VENTAS:")
#     for mes, fila in top_meses.iterrows():
#         print(f"   • {mes}: ${fila['Ingreso_Total']:,.0f} ({fila['Cantidad_Pedidos']} pedidos)")
    
#     # Trimestre con mayores ventas
#     top_trimestre = ventas_trimestral.nlargest(1, 'Ingreso_Total')
#     print(f"\n   TRIMESTRE CON MAYORES VENTAS:")
#     for trim, fila in top_trimestre.iterrows():
#         print(f"   • {trim}: ${fila['Ingreso_Total']:,.0f}")
    
#     # Días con mayores ventas
#     top_dias = ventas_diarias.nlargest(2, 'Ingreso_Total')
#     print(f"\n   DÍAS CON MAYORES VENTAS:")
#     for dia, fila in top_dias.iterrows():
#         print(f"   • {dia}: ${fila['Ingreso_Total']:,.0f}")
    
#     # 6. Visualizaciones
#     print("\n6. VISUALIZACIONES GUARDADAS:")
    
#     # Gráfico 1: Ventas mensuales (línea de tiempo)
#     plt.figure(figsize=(12, 6))
    
#     # Crear serie temporal
#     df['OrderMonthYear'] = df['OrderDate'].dt.to_period('M')
#     ventas_mensual_detalle = df.groupby('OrderMonthYear')['TotalAmount'].sum().reset_index()
#     ventas_mensual_detalle['OrderMonthYear'] = ventas_mensual_detalle['OrderMonthYear'].dt.to_timestamp()
    
#     plt.plot(ventas_mensual_detalle['OrderMonthYear'], ventas_mensual_detalle['TotalAmount'],
#             marker='o', linewidth=2, color='royalblue', markersize=6)
    
#     plt.xlabel('Fecha', fontsize=12)
#     plt.ylabel('Ingresos Totales ($)', fontsize=12)
#     plt.title('Evolución de Ventas Mensuales', fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.xticks(rotation=45)
    
#     # Añadir línea de tendencia
#     x_numeric = np.arange(len(ventas_mensual_detalle))
#     z = np.polyfit(x_numeric, ventas_mensual_detalle['TotalAmount'], 1)
#     p = np.poly1d(z)
#     plt.plot(ventas_mensual_detalle['OrderMonthYear'], p(x_numeric), 
#             "r--", alpha=0.7, label='Tendencia')
    
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'{INSIGHTS_FOLDER}/evolucion_ventas.png', dpi=300, bbox_inches='tight')
#     print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/evolucion_ventas.png'")
    
#     # Gráfico 2: Comparativa mensual (heatmap por año)
#     if len(df['OrderYear'].unique()) > 1:
#         plt.figure(figsize=(12, 8))
        
#         # Crear pivot table: años x meses
#         df['OrderMonthNum'] = df['OrderDate'].dt.month
#         heatmap_data = df.pivot_table(
#             values='TotalAmount',
#             index='OrderMonthNum',
#             columns='OrderYear',
#             aggfunc='sum'
#         ).fillna(0)
        
#         # Mapear números de mes a nombres
#         heatmap_data.index = heatmap_data.index.map(meses_nombres)
        
#         sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
#                    linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Ingresos ($)'})
        
#         plt.title('Heatmap de Ventas: Meses vs Años', fontsize=14, fontweight='bold')
#         plt.xlabel('Año', fontsize=12)
#         plt.ylabel('Mes', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f'{INSIGHTS_FOLDER}/heatmap_ventas.png', dpi=300, bbox_inches='tight')
#         print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/heatmap_ventas.png'")
    
#     # Gráfico 3: Ventas por día de la semana
#     plt.figure(figsize=(10, 6))
    
#     bars = plt.bar(ventas_diarias.index, ventas_diarias['Ingreso_Total'],
#                   color='lightgreen', edgecolor='black')
    
#     plt.xlabel('Día de la Semana', fontsize=12)
#     plt.ylabel('Ingresos Totales ($)', fontsize=12)
#     plt.title('Ventas por Día de la Semana', fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3, axis='y')
    
#     # Añadir valores en las barras
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + max(ventas_diarias['Ingreso_Total']) * 0.01,
#                 f'${height:,.0f}', ha='center', va='bottom', fontsize=10)
    
#     plt.tight_layout()
#     plt.savefig(f'{INSIGHTS_FOLDER}/ventas_dia_semana.png', dpi=300, bbox_inches='tight')
#     print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/ventas_dia_semana.png'")
#     plt.close('all')  # Cerrar todas las figuras para liberar memoria
    
#     # 7. Recomendaciones de planificación
#     print("\n7. RECOMENDACIONES DE PLANIFICACIÓN:")
    
#     print("   📊 OPTIMIZACIÓN DE INVENTARIO:")
#     print("   • Aumentar stock 30% antes de los meses pico identificados")
#     print("   • Reducir stock en meses de baja demanda para liberar capital")
    
#     print("\n   🎯 PLANIFICACIÓN DE MARKETING:")
#     print("   • Programar campañas principales 2-3 meses antes de temporada alta")
#     print("   • Crear promociones específicas para días de menor venta")
#     print("   • Ajustar presupuesto de marketing según estacionalidad")
    
#     print("\n   👥 PLANIFICACIÓN DE PERSONAL:")
#     print("   • Aumentar personal en meses de alta demanda")
#     print("   • Programar capacitaciones en meses de baja actividad")
#     print("   • Planificar vacaciones fuera de temporada alta")
    
#     print("\n   💰 PLANIFICACIÓN FINANCIERA:")
#     print("   • Anticipar flujo de caja según patrones estacionales")
#     print("   • Reservar capital para inversiones antes de temporada alta")
#     print("   • Negociar plazos de pago con proveedores según ciclos")
    
#     return ventas_anual, ventas_mensual, ventas_diarias

# # ============================================================================
# # EJECUCIÓN PRINCIPAL DE LOS NUEVOS ANÁLISIS
# # ============================================================================

# print("\n" + "="*80)
# print("EJECUTANDO ANÁLISIS AVANZADOS PARA INSIGHTS DE VALOR")
# print("="*80)

# # Ejecutar Insight 1: Productos más rentables
# productos_rentables, productos_80 = analisis_productos_rentables(df)

# # Ejecutar Insight 3: Segmentación de clientes
# segmentos_clientes = segmentacion_clientes(df, n_clusters=4)

# # Ejecutar Insight 4: Análisis temporal/estacionalidad
# ventas_anual, ventas_mensual, ventas_diarias = analisis_estacionalidad(df)

# # ============================================================================
# # RESUMEN EJECUTIVO FINAL
# # ============================================================================

# print("\n" + "="*80)
# print("RESUMEN EJECUTIVO - INSIGHTS PRINCIPALES")
# print("="*80)

# print(f"\n📁 CARPETA DE INSIGHTS: '{INSIGHTS_FOLDER}/'")
# print("-" * 50)

# print("\n🎯 INSIGHT 1: PRODUCTOS MÁS RENTABLES (80/20)")
# print("-" * 50)
# print("• Identifica qué productos generan el 80% de los ingresos")
# print("• Enfocar recursos en el 20% de productos más rentables")
# print(f"• Archivos: '{INSIGHTS_FOLDER}/top_10_productos.png', '{INSIGHTS_FOLDER}/analisis_pareto.png'")

# print("\n👥 INSIGHT 3: SEGMENTACIÓN DE CLIENTES")
# print("-" * 50)
# print("• 4 segmentos identificados: Premium, Frecuentes, Ocasionales, Inactivos")
# print("• Estrategias personalizadas para cada segmento")
# print(f"• Archivos: '{INSIGHTS_FOLDER}/distribucion_segmentos.png', '{INSIGHTS_FOLDER}/radar_segmentos.png'")

# print("\n📅 INSIGHT 4: ANÁLISIS TEMPORAL/ESTACIONALIDAD")
# print("-" * 50)
# print("• Identificación de meses/trimestres/días de mayor venta")
# print("• Optimización de inventario, marketing y personal")
# print(f"• Archivos: '{INSIGHTS_FOLDER}/evolucion_ventas.png', '{INSIGHTS_FOLDER}/ventas_dia_semana.png'")

# print("\n📋 ARCHIVOS GENERADOS EN LA CARPETA DE INSIGHTS:")
# print("-" * 50)
# archivos_insights = [
#     "top_10_productos.png         - Top 10 productos por ingresos",
#     "analisis_pareto.png          - Análisis 80/20 de productos",
#     "distribucion_segmentos.png   - Distribución de clientes por segmento",
#     "radar_segmentos.png          - Características por segmento (radar chart)",
#     "evolucion_ventas.png         - Evolución mensual de ventas",
#     "ventas_dia_semana.png        - Ventas por día de la semana"
# ]

# if len(df['OrderYear'].unique()) > 1:
#     archivos_insights.append("heatmap_ventas.png           - Heatmap ventas por mes y año")

# for archivo in archivos_insights:
#     print(f"• {archivo}")

# print(f"\n📍 RUTA COMPLETA: {os.path.abspath(INSIGHTS_FOLDER)}/")

# print("\n" + "="*80)
# print("ANÁLISIS COMPLETADO - LISTO PARA PRESENTACIÓN")
# print("9. K-MEANS SOBRE COMPONENTES PRINCIPALES")
# print("="*80)

# inertias = []
# K = range(1, 11)

# for k in K:
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     km.fit(X_pca)
#     inertias.append(km.inertia_)

# plt.figure(figsize=(7,5))
# plt.plot(K, inertias, marker='o')
# plt.xlabel("Número de clusters (k)")
# plt.ylabel("Inercia")
# plt.title("Método del Codo")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("elbow_kmeans.png", dpi=300)
# plt.close()
# kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
# df["cluster_kmeans"] = kmeans.fit_predict(X_pca)

# centroides = kmeans.cluster_centers_

# plt.figure(figsize=(8,6))
# plt.scatter(
#     X_pca[:,0], X_pca[:,1],
#     c=df["cluster_kmeans"],
#     cmap="tab10",
#     s=50
# )
# plt.scatter(
#     centroides[:,0], centroides[:,1],
#     c="red", s=200, marker="X", label="Centroides"
# )
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Clusters K-Means sobre PCA")
# plt.legend()
# plt.tight_layout()
# plt.savefig("kmeans_pca.png", dpi=300)
# plt.close()

# print("\n" + "="*80)
# print("10. ANÁLISIS DE CORRELACIÓN")
# print("="*80)

# corr_matrix = df_scaled.corr()

# plt.figure(figsize=(10,8))
# sns.heatmap(
#     corr_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="vlag",
#     center=0,
#     square=True,
#     linewidths=0.5
# )
# plt.title("Matriz de Correlación - Variables Numéricas")
# plt.tight_layout()
# plt.savefig("matriz_correlacion.png", dpi=300)
# plt.close()

# print("Gráfico guardado como 'matriz_correlacion.png'")

# print("\n" + "="*80)
# print("8.x CARGAS FACTORIALES (Contribución de variables al PCA)")
# print("="*80)

# # DataFrame de cargas factoriales
# loadings = pd.DataFrame(
#     pca.components_.T,
#     columns=[f"PC{i+1}" for i in range(pca.n_components_)],
#     index=df_scaled.columns
# )

# print("\nCargas factoriales (primeras filas):")
# print(loadings.head())

# n_top = 10  # número de variables a mostrar

# for i in range(min(3, pca.n_components_)):
#     pc = f"PC{i+1}"
#     print(f"\nVariables más influyentes en {pc}:")
    
#     top_vars = loadings[pc].abs().sort_values(ascending=False).head(n_top)
#     for var in top_vars.index:
#         value = loadings.loc[var, pc]
#         direction = "positiva" if value > 0 else "negativa"
#         print(f"  • {var}: {value:.3f} ({direction})")

# plt.figure(figsize=(8,6))

# top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(10).index
# sns.barplot(
#     x=loadings.loc[top_pc1, "PC1"],
#     y=top_pc1,
#     color="steelblue"
# )

# plt.title("Cargas factoriales - PC1")
# plt.xlabel("Carga")
# plt.ylabel("Variable")
# plt.axvline(0, color='black', linewidth=0.8)
# plt.tight_layout()
# plt.savefig("cargas_pc1.png", dpi=300)
# plt.close()

# if pca.n_components_ >= 2:
#     plt.figure(figsize=(8,6))
    
#     top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(10).index
#     sns.barplot(
#         x=loadings.loc[top_pc2, "PC2"],
#         y=top_pc2,
#         color="darkorange"
#     )

#     plt.title("Cargas factoriales - PC2")
#     plt.xlabel("Carga")
#     plt.ylabel("Variable")
#     plt.axvline(0, color='black', linewidth=0.8)
#     plt.tight_layout()
#     plt.savefig("cargas_pc2.png", dpi=300)
#     plt.close()
