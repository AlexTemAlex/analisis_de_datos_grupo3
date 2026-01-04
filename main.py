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
df = pd.read_csv(DATASET)

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
