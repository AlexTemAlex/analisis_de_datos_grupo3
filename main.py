import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

DATASET = "fmcg_sales_3years_1M_rows.csv"

pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.width', 150)         # Ajusta el ancho de la salida

dtypes = {
    'year': 'Int16',
    'month': 'Int16',
    'day': 'Int16',
    'weekofyear': 'Int16',
    'weekday': 'Int16',
    'is_weekend': 'category',
    'is_holiday': 'category',
    'temperature': 'float32',
    'rain_mm': 'float32',
    'store_id': 'category',
    'country': 'category',
    'city': 'category',
    'channel': 'category',
    'sku_id': 'category',
    'sku_name': 'category',
    'category': 'category',
    'subcategory': 'category',
    'brand': 'category',
    'units_sold': 'Int16',
    'list_price': 'float32',
    'discount_pct': 'float32',
    'promo_flag': 'category',
    'gross_sales': 'float32',
    'net_sales': 'float32',
    'stock_on_hand': 'Int16',
    'stock_out_flag': 'category',
    'lead_time_days': 'Int16',
    'supplier_id': 'category',
    'purchase_cost': 'float32',
    'margin_pct': 'float32'
}

df = pd.read_csv(DATASET,
                usecols=list(dtypes.keys()),  # solo columnas de dtypes
                dtype=dtypes)
                # nrows=5) 

print(f"*****Data Set*****\n{df}\n")
print(f"*****Tipos de Variables Definidas*****")
print(f"{df.info()}\n")

print(f"*****Promedio de Valores Nulos*****\n{df.isnull().mean().sort_values(ascending=False)}\n")

print(f"*****Cantidad de Valores Duplicados*****\n{df.duplicated(subset=['year','month','day','store_id','sku_id']).sum()}\n")


#Especificacion de variables numericas y categoricas
num_real = [
    'temperature', 'rain_mm', 'units_sold',
    'list_price', 'discount_pct', 'gross_sales', 'net_sales',
    'stock_on_hand', 'lead_time_days', 'purchase_cost',
    'margin_pct'
    
]

flags = [
    'is_weekend', 'is_holiday',
    'promo_flag', 'stock_out_flag'
]

categ = [
    'year', 'month', 'day', 'weekofyear', 'weekday',
    'store_id', 'country', 'city', 'channel', 'sku_id',
    'sku_name', 'category', 'subcategory', 'brand', 'supplier_id'
]

print(f"*****Etadisticas Básicas - Variables Numéricas*****\n{df[num_real].describe().T}\n")
print(f"*****Etadisticas Básicas - Variables Numéricas Binarias (Codigo Disyuntivo)*****\n{df[flags].describe(include="category").T}\n")
print(f"*****Etadisticas Básicas - Variables Categóricas***** \n{df[categ].describe(include="category").T}\n")

print(f"*****Centrar y Reducir*****")
df_num = df[num_real].dropna() # Eliminar Valores Nulos
scaler = StandardScaler()

df_scaled = pd.DataFrame(
    scaler.fit_transform(df_num),
    columns=num_real,
    index=df_num.index
)

print(f"\n{df_scaled.describe().loc[['mean','std']]}\n")