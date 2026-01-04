import matplotlib
# Usar backend no interactivo para evitar problemas con Tkinter
matplotlib.use('Agg')  

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os  # Importar módulo os para manejo de carpetas

DATASET = "Amazon.csv"

# --- CREAR CARPETA PARA INSIGHTS ---
INSIGHTS_FOLDER = "insights"
if not os.path.exists(INSIGHTS_FOLDER):
    os.makedirs(INSIGHTS_FOLDER)
    print(f"\n📁 Carpeta '{INSIGHTS_FOLDER}' creada para guardar gráficos de insights")

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
print("ANÁLISIS BÁSICO COMPLETADO")
print("="*80)

# ============================================================================
# NUEVO PUNTO 12: ANÁLISIS DE PRODUCTOS MÁS RENTABLES (INSIGHT 1)
# ============================================================================

def analisis_productos_rentables(df):
    """
    Insight 1: Análisis de productos más rentables (Regla 80/20)
    """
    print("\n" + "="*80)
    print("12. ANÁLISIS DE PRODUCTOS MÁS RENTABLES - REGLA 80/20")
    print("="*80)
    
    # 1. Productos por ingresos totales
    print("\n1. TOP 10 PRODUCTOS POR INGRESOS TOTALES:")
    productos_ingresos = df.groupby('ProductName')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
    productos_ingresos = productos_ingresos.rename(columns={
        'sum': 'Ingreso_Total',
        'count': 'Cantidad_Ventas',
        'mean': 'Ticket_Promedio'
    }).sort_values('Ingreso_Total', ascending=False)
    
    print(productos_ingresos.head(10))
    
    # 2. Análisis 80/20 (Pareto)
    print("\n2. ANÁLISIS PARETO (80/20):")
    productos_ingresos_sorted = productos_ingresos.sort_values('Ingreso_Total', ascending=False)
    productos_ingresos_sorted['Ingreso_Acumulado'] = productos_ingresos_sorted['Ingreso_Total'].cumsum()
    productos_ingresos_sorted['%_Acumulado'] = (productos_ingresos_sorted['Ingreso_Acumulado'] / 
                                                productos_ingresos_sorted['Ingreso_Total'].sum() * 100)
    
    # Encontrar qué productos generan el 80% de ingresos
    productos_80 = productos_ingresos_sorted[productos_ingresos_sorted['%_Acumulado'] <= 80]
    n_productos_80 = len(productos_80)
    total_productos = len(productos_ingresos_sorted)
    porcentaje_productos = (n_productos_80 / total_productos) * 100
    
    print(f"   • Total productos: {total_productos}")
    print(f"   • Productos que generan 80% de ingresos: {n_productos_80}")
    print(f"   • Esto representa el {porcentaje_productos:.1f}% de todos los productos")
    print(f"   • {100 - porcentaje_productos:.1f}% de productos generan solo 20% de ingresos")
    
    # Mostrar los productos clave
    print(f"\n   PRODUCTOS CLAVE (generan 80% de ingresos):")
    for i, (producto, fila) in enumerate(productos_80.head(15).iterrows(), 1):
        print(f"   {i:2d}. {producto[:40]:40s} | ${fila['Ingreso_Total']:>10,.0f} | {fila['Cantidad_Ventas']:>4} ventas")
    
    # 3. Categorías más rentables
    print("\n3. CATEGORÍAS MÁS RENTABLES:")
    categorias_ingresos = df.groupby('Category')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
    categorias_ingresos = categorias_ingresos.rename(columns={
        'sum': 'Ingreso_Total',
        'count': 'Cantidad_Ventas',
        'mean': 'Ticket_Promedio'
    }).sort_values('Ingreso_Total', ascending=False)
    
    print(categorias_ingresos)
    
    # 4. Marcas más rentables
    print("\n4. MARCAS MÁS RENTABLES:")
    marcas_ingresos = df.groupby('Brand')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
    marcas_ingresos = marcas_ingresos.rename(columns={
        'sum': 'Ingreso_Total',
        'count': 'Cantidad_Ventas',
        'mean': 'Ticket_Promedio'
    }).sort_values('Ingreso_Total', ascending=False)
    
    print(marcas_ingresos.head(10))
    
    # 5. Visualizaciones
    print("\n5. VISUALIZACIONES GUARDADAS:")
    
    # Gráfico 1: Top 10 productos por ingresos
    plt.figure(figsize=(12, 6))
    top_10_productos = productos_ingresos.head(10)
    bars = plt.barh(range(len(top_10_productos)), top_10_productos['Ingreso_Total'], 
                   color='skyblue', edgecolor='black')
    plt.yticks(range(len(top_10_productos)), top_10_productos.index, fontsize=9)
    plt.xlabel('Ingreso Total ($)', fontsize=12)
    plt.title('Top 10 Productos por Ingresos Totales', fontsize=14, fontweight='bold')
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(top_10_productos['Ingreso_Total']) * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'${width:,.0f}', ha='left', va='center', fontsize=9)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{INSIGHTS_FOLDER}/top_10_productos.png', dpi=300, bbox_inches='tight')
    print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/top_10_productos.png'")
    
    # Gráfico 2: Análisis Pareto
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Barras de ingresos
    ax1.bar(range(len(productos_ingresos_sorted.head(20))), 
           productos_ingresos_sorted.head(20)['Ingreso_Total'],
           color='lightblue', alpha=0.7, label='Ingreso por Producto')
    ax1.set_xlabel('Productos (ordenados por ingresos)')
    ax1.set_ylabel('Ingreso Total ($)', color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')
    
    # Línea de Pareto
    ax2 = ax1.twinx()
    ax2.plot(range(len(productos_ingresos_sorted.head(20))),
            productos_ingresos_sorted.head(20)['%_Acumulado'],
            color='red', marker='o', linewidth=2, markersize=4,
            label='% Acumulado')
    ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80%')
    ax2.set_ylabel('% Ingreso Acumulado', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)
    
    plt.title('Análisis Pareto - Ingresos por Producto', fontsize=14, fontweight='bold')
    plt.xticks(range(len(productos_ingresos_sorted.head(20))), 
              [f'P{i+1}' for i in range(20)], rotation=45)
    
    # Combinar leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{INSIGHTS_FOLDER}/analisis_pareto.png', dpi=300, bbox_inches='tight')
    print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/analisis_pareto.png'")
    plt.close('all')  # Cerrar todas las figuras para liberar memoria
    
    # 6. Recomendaciones específicas
    print("\n6. RECOMENDACIONES DE NEGOCIO:")
    print("   ✓ ENFOCAR INVENTARIO en los productos del top 20% (ver archivo 'analisis_pareto.png')")
    print("   ✓ CREAR BUNDLES con productos complementarios de alta rentabilidad")
    print("   ✓ NEGOCIAR MEJORES CONDICIONES con marcas del top 10 (ver análisis de marcas)")
    print("   ✓ OPTIMIZAR PRECIOS en categorías con mayor ticket promedio")
    print("   ✓ DESARROLLAR CAMPOS DE VENTAS cruzadas entre productos del mismo cliente")
    
    return productos_ingresos, productos_80

# ============================================================================
# NUEVO PUNTO 13: SEGMENTACIÓN DE CLIENTES (INSIGHT 3)
# ============================================================================

def segmentacion_clientes(df, n_clusters=4):
    """
    Insight 3: Segmentación de clientes en 4 grupos basados en comportamiento
    """
    print("\n" + "="*80)
    print("13. SEGMENTACIÓN DE CLIENTES - 4 GRUPOS DE COMPORTAMIENTO")
    print("="*80)
    
    # 1. Preparar datos de clientes
    print("\n1. PREPARACIÓN DE DATOS DE CLIENTES:")
    
    # Calcular métricas por cliente
    clientes_agg = df.groupby('CustomerID').agg({
        'TotalAmount': ['sum', 'mean', 'count'],  # Ingreso total, ticket promedio, frecuencia
        'Quantity': 'sum',                        # Cantidad total comprada
        'Discount': 'mean',                       # Sensibilidad a descuentos
        'UnitPrice': 'mean',                      # Precio promedio pagado
        'OrderDate': 'max'                        # Última compra
    }).round(2)
    
    # Renombrar columnas
    clientes_agg.columns = ['Ingreso_Total', 'Ticket_Promedio', 'Frecuencia',
                           'Cantidad_Total', 'Descuento_Promedio', 
                           'Precio_Promedio', 'Ultima_Compra']
    
    # Calcular días desde última compra
    clientes_agg['Dias_Ultima_Compra'] = (pd.Timestamp.now() - clientes_agg['Ultima_Compra']).dt.days
    
    print(f"   • Total clientes analizados: {len(clientes_agg)}")
    print(f"   • Métricas calculadas: Ingreso total, Ticket promedio, Frecuencia, etc.")
    
    # 2. Estandarizar datos para clustering
    print("\n2. APLICACIÓN DE CLUSTERING (K-Means):")
    
    # Seleccionar variables para clustering
    clustering_vars = ['Ingreso_Total', 'Ticket_Promedio', 'Frecuencia', 
                      'Descuento_Promedio', 'Dias_Ultima_Compra']
    
    # Estandarizar
    scaler = StandardScaler()
    clientes_scaled = pd.DataFrame(
        scaler.fit_transform(clientes_agg[clustering_vars].fillna(0)),
        columns=clustering_vars,
        index=clientes_agg.index
    )
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clientes_agg['Segmento'] = kmeans.fit_predict(clientes_scaled)
    
    # 3. Analizar segmentos
    print(f"\n3. CARACTERÍSTICAS DE LOS {n_clusters} SEGMENTOS:")
    
    segmentos_analisis = clientes_agg.groupby('Segmento')[clustering_vars].agg(['mean', 'count']).round(2)
    
    # Renombrar segmentos según características
    segmento_nombres = {
        0: '🎯 PREMIUM (Alto Valor)',
        1: '🔄 FRECUENTES (Leales)',
        2: '💰 OCASIONALES (Sensibles Precio)',
        3: '⏰ INACTIVOS (Riesgo Pérdida)'
    }
    
    clientes_agg['Segmento_Nombre'] = clientes_agg['Segmento'].map(segmento_nombres)
    
    print("\n   RESUMEN POR SEGMENTO:")
    for seg_num, seg_nombre in segmento_nombres.items():
        seg_data = clientes_agg[clientes_agg['Segmento'] == seg_num]
        print(f"\n   {seg_nombre}:")
        print(f"     • Número de clientes: {len(seg_data)}")
        print(f"     • Ingreso total promedio: ${seg_data['Ingreso_Total'].mean():,.0f}")
        print(f"     • Ticket promedio: ${seg_data['Ticket_Promedio'].mean():,.0f}")
        print(f"     • Frecuencia promedio: {seg_data['Frecuencia'].mean():.1f} compras")
        print(f"     • Días desde última compra: {seg_data['Dias_Ultima_Compra'].mean():.0f} días")
    
    # 4. Top clientes por segmento
    print("\n4. TOP 5 CLIENTES POR SEGMENTO:")
    
    for seg_num, seg_nombre in segmento_nombres.items():
        seg_clientes = clientes_agg[clientes_agg['Segmento'] == seg_num]
        top_5 = seg_clientes.nlargest(5, 'Ingreso_Total')
        
        print(f"\n   {seg_nombre}:")
        for idx, (cliente_id, fila) in enumerate(top_5.iterrows(), 1):
            print(f"     {idx}. Cliente {cliente_id}:")
            print(f"        - Ingreso total: ${fila['Ingreso_Total']:,.0f}")
            print(f"        - Ticket promedio: ${fila['Ticket_Promedio']:,.0f}")
            print(f"        - Frecuencia: {fila['Frecuencia']:.0f} compras")
    
    # 5. Visualizaciones
    print("\n5. VISUALIZACIONES GUARDADAS:")
    
    # Gráfico 1: Distribución de segmentos
    plt.figure(figsize=(10, 6))
    segment_counts = clientes_agg['Segmento_Nombre'].value_counts()
    colors = ['gold', 'lightgreen', 'lightcoral', 'lightblue']
    
    plt.pie(segment_counts.values, labels=segment_counts.index,
           autopct='%1.1f%%', startangle=90, colors=colors,
           wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    plt.title('Distribución de Clientes por Segmento', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{INSIGHTS_FOLDER}/distribucion_segmentos.png', dpi=300, bbox_inches='tight')
    print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/distribucion_segmentos.png'")
    
    # Gráfico 2: Características por segmento (radar chart)
    fig = plt.figure(figsize=(10, 8))
    
    # Preparar datos para radar chart
    segment_means = clientes_agg.groupby('Segmento_Nombre')[clustering_vars].mean()
    
    # Normalizar para radar chart
    segment_normalized = segment_means.copy()
    for col in clustering_vars:
        segment_normalized[col] = (segment_means[col] - segment_means[col].min()) / \
                                 (segment_means[col].max() - segment_means[col].min())
    
    # Crear radar chart
    angles = np.linspace(0, 2 * np.pi, len(clustering_vars), endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo
    
    ax = fig.add_subplot(111, polar=True)
    
    for idx, (seg_name, seg_data) in enumerate(segment_normalized.iterrows()):
        values = seg_data.tolist()
        values += values[:1]  # Cerrar el círculo
        
        ax.plot(angles, values, 'o-', linewidth=2, label=seg_name)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(clustering_vars, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Características por Segmento de Clientes', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(f'{INSIGHTS_FOLDER}/radar_segmentos.png', dpi=300, bbox_inches='tight')
    print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/radar_segmentos.png'")
    plt.close('all')  # Cerrar todas las figuras para liberar memoria
    
    # 6. Recomendaciones por segmento
    print("\n6. ESTRATEGIAS POR SEGMENTO:")
    
    estrategias = {
        '🎯 PREMIUM (Alto Valor)': [
            "✓ Programa de fidelización premium",
            "✓ Atención personalizada (asesor dedicado)",
            "✓ Acceso anticipado a nuevos productos",
            "✓ Eventos exclusivos para el segmento"
        ],
        '🔄 FRECUENTES (Leales)': [
            "✓ Programa de puntos por compras",
            "✓ Descuentos por volumen/repetición",
            "✓ Recomendaciones personalizadas",
            "✓ Encuestas de satisfacción periódicas"
        ],
        '💰 OCASIONALES (Sensibles Precio)': [
            "✓ Ofertas y promociones específicas",
            "✓ Recordatorios de carrito abandonado",
            "✓ Comparativas de precio vs competencia",
            "✓ Programas de referidos con incentivos"
        ],
        '⏰ INACTIVOS (Riesgo Pérdida)': [
            "✓ Campañas de reactivación (email/SMS)",
            "✓ Ofertas de re-enganche",
            "✓ Encuestas para entender causas",
            "✓ Programa de win-back específico"
        ]
    }
    
    for segmento, acciones in estrategias.items():
        print(f"\n   {segmento}:")
        for accion in acciones:
            print(f"   {accion}")
    
    return clientes_agg

# ============================================================================
# NUEVO PUNTO 12: ANÁLISIS TEMPORAL/ESTACIONALIDAD (INSIGHT 4)
# ============================================================================

def analisis_estacionalidad(df):
    """
    Insight 4: Análisis temporal y estacionalidad
    """
    print("\n" + "="*80)
    print("14. ANÁLISIS TEMPORAL Y ESTACIONALIDAD")
    print("="*80)
    
    # 1. Ventas por año
    print("\n1. VENTAS POR AÑO:")
    ventas_anual = df.groupby('OrderYear')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
    ventas_anual = ventas_anual.rename(columns={
        'sum': 'Ingreso_Total',
        'count': 'Cantidad_Pedidos',
        'mean': 'Ticket_Promedio'
    })
    
    print(ventas_anual)
    
    # Cálculo de crecimiento anual
    if len(ventas_anual) > 1:
        print("\n   CRECIMIENTO ANUAL:")
        ventas_anual['Crecimiento_%'] = ventas_anual['Ingreso_Total'].pct_change() * 100
        print(ventas_anual[['Ingreso_Total', 'Crecimiento_%']].round(2))
    
    # 2. Ventas por mes (promedio)
    print("\n2. VENTAS POR MES (PROMEDIO):")
    
    # Mapeo de números de mes a nombres
    meses_nombres = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    
    ventas_mensual = df.groupby('OrderMonth')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
    ventas_mensual = ventas_mensual.rename(columns={
        'sum': 'Ingreso_Total',
        'count': 'Cantidad_Pedidos',
        'mean': 'Ticket_Promedio'
    })
    ventas_mensual.index = ventas_mensual.index.map(meses_nombres)
    
    print(ventas_mensual)
    
    # 3. Ventas por trimestre
    print("\n3. VENTAS POR TRIMESTRE:")
    trimestre_nombres = {1: 'Q1 (Ene-Mar)', 2: 'Q2 (Abr-Jun)', 
                        3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dic)'}
    
    ventas_trimestral = df.groupby('OrderQuarter')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
    ventas_trimestral = ventas_trimestral.rename(columns={
        'sum': 'Ingreso_Total',
        'count': 'Cantidad_Pedidos',
        'mean': 'Ticket_Promedio'
    })
    ventas_trimestral.index = ventas_trimestral.index.map(trimestre_nombres)
    
    print(ventas_trimestral)
    
    # 4. Días de la semana con más ventas
    print("\n4. VENTAS POR DÍA DE LA SEMANA:")
    df['Dia_Semana'] = df['OrderDate'].dt.day_name()
    
    # Ordenar días de la semana
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                 'Friday', 'Saturday', 'Sunday']
    dias_espanol = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    
    ventas_diarias = df.groupby('Dia_Semana')['TotalAmount'].agg(['sum', 'count', 'mean']).round(2)
    ventas_diarias = ventas_diarias.rename(columns={
        'sum': 'Ingreso_Total',
        'count': 'Cantidad_Pedidos',
        'mean': 'Ticket_Promedio'
    })
    
    # Reindexar para orden correcto
    ventas_diarias = ventas_diarias.reindex(dias_orden)
    ventas_diarias.index = ventas_diarias.index.map(dias_espanol)
    
    print(ventas_diarias)
    
    # 5. Identificar temporadas pico
    print("\n5. IDENTIFICACIÓN DE TEMPORADAS PICO:")
    
    # Meses con mayores ventas (top 3)
    top_meses = ventas_mensual.nlargest(3, 'Ingreso_Total')
    print("   MESES CON MAYORES VENTAS:")
    for mes, fila in top_meses.iterrows():
        print(f"   • {mes}: ${fila['Ingreso_Total']:,.0f} ({fila['Cantidad_Pedidos']} pedidos)")
    
    # Trimestre con mayores ventas
    top_trimestre = ventas_trimestral.nlargest(1, 'Ingreso_Total')
    print(f"\n   TRIMESTRE CON MAYORES VENTAS:")
    for trim, fila in top_trimestre.iterrows():
        print(f"   • {trim}: ${fila['Ingreso_Total']:,.0f}")
    
    # Días con mayores ventas
    top_dias = ventas_diarias.nlargest(2, 'Ingreso_Total')
    print(f"\n   DÍAS CON MAYORES VENTAS:")
    for dia, fila in top_dias.iterrows():
        print(f"   • {dia}: ${fila['Ingreso_Total']:,.0f}")
    
    # 6. Visualizaciones
    print("\n6. VISUALIZACIONES GUARDADAS:")
    
    # Gráfico 1: Ventas mensuales (línea de tiempo)
    plt.figure(figsize=(12, 6))
    
    # Crear serie temporal
    df['OrderMonthYear'] = df['OrderDate'].dt.to_period('M')
    ventas_mensual_detalle = df.groupby('OrderMonthYear')['TotalAmount'].sum().reset_index()
    ventas_mensual_detalle['OrderMonthYear'] = ventas_mensual_detalle['OrderMonthYear'].dt.to_timestamp()
    
    plt.plot(ventas_mensual_detalle['OrderMonthYear'], ventas_mensual_detalle['TotalAmount'],
            marker='o', linewidth=2, color='royalblue', markersize=6)
    
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Ingresos Totales ($)', fontsize=12)
    plt.title('Evolución de Ventas Mensuales', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Añadir línea de tendencia
    x_numeric = np.arange(len(ventas_mensual_detalle))
    z = np.polyfit(x_numeric, ventas_mensual_detalle['TotalAmount'], 1)
    p = np.poly1d(z)
    plt.plot(ventas_mensual_detalle['OrderMonthYear'], p(x_numeric), 
            "r--", alpha=0.7, label='Tendencia')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{INSIGHTS_FOLDER}/evolucion_ventas.png', dpi=300, bbox_inches='tight')
    print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/evolucion_ventas.png'")
    
    # Gráfico 2: Comparativa mensual (heatmap por año)
    if len(df['OrderYear'].unique()) > 1:
        plt.figure(figsize=(12, 8))
        
        # Crear pivot table: años x meses
        df['OrderMonthNum'] = df['OrderDate'].dt.month
        heatmap_data = df.pivot_table(
            values='TotalAmount',
            index='OrderMonthNum',
            columns='OrderYear',
            aggfunc='sum'
        ).fillna(0)
        
        # Mapear números de mes a nombres
        heatmap_data.index = heatmap_data.index.map(meses_nombres)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                   linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Ingresos ($)'})
        
        plt.title('Heatmap de Ventas: Meses vs Años', fontsize=14, fontweight='bold')
        plt.xlabel('Año', fontsize=12)
        plt.ylabel('Mes', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{INSIGHTS_FOLDER}/heatmap_ventas.png', dpi=300, bbox_inches='tight')
        print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/heatmap_ventas.png'")
    
    # Gráfico 3: Ventas por día de la semana
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(ventas_diarias.index, ventas_diarias['Ingreso_Total'],
                  color='lightgreen', edgecolor='black')
    
    plt.xlabel('Día de la Semana', fontsize=12)
    plt.ylabel('Ingresos Totales ($)', fontsize=12)
    plt.title('Ventas por Día de la Semana', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(ventas_diarias['Ingreso_Total']) * 0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{INSIGHTS_FOLDER}/ventas_dia_semana.png', dpi=300, bbox_inches='tight')
    print(f"   • Gráfico guardado: '{INSIGHTS_FOLDER}/ventas_dia_semana.png'")
    plt.close('all')  # Cerrar todas las figuras para liberar memoria
    
    # 7. Recomendaciones de planificación
    print("\n7. RECOMENDACIONES DE PLANIFICACIÓN:")
    
    print("   📊 OPTIMIZACIÓN DE INVENTARIO:")
    print("   • Aumentar stock 30% antes de los meses pico identificados")
    print("   • Reducir stock en meses de baja demanda para liberar capital")
    
    print("\n   🎯 PLANIFICACIÓN DE MARKETING:")
    print("   • Programar campañas principales 2-3 meses antes de temporada alta")
    print("   • Crear promociones específicas para días de menor venta")
    print("   • Ajustar presupuesto de marketing según estacionalidad")
    
    print("\n   👥 PLANIFICACIÓN DE PERSONAL:")
    print("   • Aumentar personal en meses de alta demanda")
    print("   • Programar capacitaciones en meses de baja actividad")
    print("   • Planificar vacaciones fuera de temporada alta")
    
    print("\n   💰 PLANIFICACIÓN FINANCIERA:")
    print("   • Anticipar flujo de caja según patrones estacionales")
    print("   • Reservar capital para inversiones antes de temporada alta")
    print("   • Negociar plazos de pago con proveedores según ciclos")
    
    return ventas_anual, ventas_mensual, ventas_diarias

# ============================================================================
# EJECUCIÓN PRINCIPAL DE LOS NUEVOS ANÁLISIS
# ============================================================================

print("\n" + "="*80)
print("EJECUTANDO ANÁLISIS AVANZADOS PARA INSIGHTS DE VALOR")
print("="*80)

# Ejecutar Insight 1: Productos más rentables
productos_rentables, productos_80 = analisis_productos_rentables(df)

# Ejecutar Insight 3: Segmentación de clientes
segmentos_clientes = segmentacion_clientes(df, n_clusters=4)

# Ejecutar Insight 4: Análisis temporal/estacionalidad
ventas_anual, ventas_mensual, ventas_diarias = analisis_estacionalidad(df)

# ============================================================================
# RESUMEN EJECUTIVO FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMEN EJECUTIVO - INSIGHTS PRINCIPALES")
print("="*80)

print(f"\n📁 CARPETA DE INSIGHTS: '{INSIGHTS_FOLDER}/'")
print("-" * 50)

print("\n🎯 INSIGHT 1: PRODUCTOS MÁS RENTABLES (80/20)")
print("-" * 50)
print("• Identifica qué productos generan el 80% de los ingresos")
print("• Enfocar recursos en el 20% de productos más rentables")
print(f"• Archivos: '{INSIGHTS_FOLDER}/top_10_productos.png', '{INSIGHTS_FOLDER}/analisis_pareto.png'")

print("\n👥 INSIGHT 3: SEGMENTACIÓN DE CLIENTES")
print("-" * 50)
print("• 4 segmentos identificados: Premium, Frecuentes, Ocasionales, Inactivos")
print("• Estrategias personalizadas para cada segmento")
print(f"• Archivos: '{INSIGHTS_FOLDER}/distribucion_segmentos.png', '{INSIGHTS_FOLDER}/radar_segmentos.png'")

print("\n📅 INSIGHT 4: ANÁLISIS TEMPORAL/ESTACIONALIDAD")
print("-" * 50)
print("• Identificación de meses/trimestres/días de mayor venta")
print("• Optimización de inventario, marketing y personal")
print(f"• Archivos: '{INSIGHTS_FOLDER}/evolucion_ventas.png', '{INSIGHTS_FOLDER}/ventas_dia_semana.png'")

print("\n📋 ARCHIVOS GENERADOS EN LA CARPETA DE INSIGHTS:")
print("-" * 50)
archivos_insights = [
    "top_10_productos.png         - Top 10 productos por ingresos",
    "analisis_pareto.png          - Análisis 80/20 de productos",
    "distribucion_segmentos.png   - Distribución de clientes por segmento",
    "radar_segmentos.png          - Características por segmento (radar chart)",
    "evolucion_ventas.png         - Evolución mensual de ventas",
    "ventas_dia_semana.png        - Ventas por día de la semana"
]

if len(df['OrderYear'].unique()) > 1:
    archivos_insights.append("heatmap_ventas.png           - Heatmap ventas por mes y año")

for archivo in archivos_insights:
    print(f"• {archivo}")

print(f"\n📍 RUTA COMPLETA: {os.path.abspath(INSIGHTS_FOLDER)}/")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO - LISTO PARA PRESENTACIÓN")
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