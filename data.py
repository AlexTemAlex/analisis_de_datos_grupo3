import pandas as pd
import random

# --- Generar 100 nombres comunes ---
nombres_usa = [
    "James","John","Robert","Michael","William","David","Richard","Joseph","Thomas","Charles",
    "Christopher","Daniel","Matthew","Anthony","Mark","Donald","Steven","Paul","Andrew","Joshua",
    "Mary","Patricia","Jennifer","Linda","Elizabeth","Barbara","Susan","Jessica","Sarah","Karen",
    "Nancy","Lisa","Margaret","Betty","Sandra","Ashley","Emily","Kimberly","Donna","Michelle",
    "Amanda","Melissa","Stephanie","Rebecca","Laura","Sharon","Cynthia","Kathleen","Amy","Angela",
    "Brian","Kevin","George","Edward","Ronald","Timothy","Jason","Jeffrey","Ryan","Jacob",
    "Nancy","Carol","Evelyn","Megan","Helen","Deborah","Rachel","Samantha","Teresa","Diana",
    "Ann","Alice","Gloria","Marie","Janet","Frances","Kathryn","Jean","Christine","Marie",
    "Marie","Denise","Theresa","Carolyn","Ruth","Shirley","Linda","Barbara","Elizabeth","Mary"
]

# --- Generar 200 apellidos comunes ---
apellidos_usa = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
    "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
    "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson",
    "Walker","Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
    "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts",
    "Gomez","Phillips","Evans","Turner","Diaz","Parker","Cruz","Edwards","Collins","Reyes",
    "Stewart","Morris","Morales","Murphy","Cook","Rogers","Gutierrez","Ortiz","Morgan","Cooper",
    "Peterson","Bailey","Reed","Kelly","Howard","Ramos","Kim","Cox","Ward","Richardson",
    "Watson","Brooks","Chavez","Wood","James","Bennett","Gray","Mendoza","Ruiz","Hughes",
    "Price","Alvarez","Castillo","Sanders","Patel","Myers","Long","Ross","Foster","Jimenez",
    "Powell","Jenkins","Perry","Russell","Sullivan","Bell","Coleman","Butler","Henderson","Barnes",
    "Gonzales","Fisher","Vasquez","Simmons","Romero","Jordan","Patterson","Alexander","Hamilton","Graham",
    "Reynolds","Griffin","Wallace","Moreno","West","Cole","Hayes","Bryant","Herrera","Gibson",
    "Ellis","Tran","Medina","Aguilar","Stevens","Murray","Ford","Castro","Marshall","Owens",
    "Harrison","Fernandez","McDonald","Woods","Washington","Kennedy","Wells","Vargas","Henry","Chen",
    "Freeman","Webb","Tucker","Guzman","Burns","Crawford","Olson","Simpson","Porter","Hunter",
    "Gordon","Mendez","Silva","Shaw","Snyder","Mason","Dixon","Munoz","Hunt","Hicks",
    "Holmes","Palmer","Wagner","Black","Robertson","Boyd","Rose","Stone","Salazar","Fox",
    "Warren","Mills","Meyer","Rice","Schmidt","Garza","Daniels","Ferguson","Nichols","Stephens",
    "Soto","Weaver","Ryan","Gardner","Payne","Grant","Dunn","Kelley","Spencer","Hawkins",
    "Arnold","Pierce","Vazquez","Hansen","Peters","Santos","Hart","Bradley","Knight","Elliott",
    "Cunningham","Duncan","Armstrong","Hudson","Carroll","Lane","Riley","Andrews","Alvarado","Ray"
]

# --- Parámetros ---
cantidad = 37823

# --- Generar IDs únicos ---
ids_clientes = random.sample(range(1, 90001), cantidad)
ids_clientes = [f"CUST{str(i).zfill(6)}" for i in ids_clientes]

# --- Generar nombres completos con 1 o 2 apellidos ---
nombres_completos = []
for _ in range(cantidad):
    nombre = random.choice(nombres_usa)
    apellidos = [random.choice(apellidos_usa)]
    # 50% de probabilidad de agregar un segundo apellido
    if random.random() < 0.5:
        segundo = random.choice(apellidos_usa)
        # evitar repetir el mismo apellido dos veces
        if segundo != apellidos[0]:
            apellidos.append(segundo)
    nombres_completos.append(f"{nombre} {' '.join(apellidos)}")

# --- Crear DataFrame ---
df_usuarios = pd.DataFrame({
    "CustomerID": ids_clientes,
    "CustomerName": nombres_completos
})


DATASET = "Amazon.csv"
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

# --- Verificación ---
print(df_usuarios.head())
print("Total usuarios:", len(df_usuarios))
print("Usuarios únicos (ID):", df_usuarios["CustomerID"].nunique())
print("Nombres únicos (aprox):", df_usuarios["CustomerName"].nunique())


# --- Cantidad de órdenes (tamaño del dataset) ---
cantidad_ordenes = len(DF)  # 100,000

# --- Distribución de tipos de clientes ---
tipos_clientes = {
    "Proveedor": 0.01,   # 1% de usuarios con muchas compras
    "Frecuente": 0.19,   # 19% clientes regulares
    "Ocasional": 0.50,   # 50% compras moderadas
    "Único": 0.30        # 30% solo una compra
}

total_usuarios = len(df_usuarios)

# --- Calcular cantidad de usuarios por tipo ---
ordenes_por_tipo = {tipo: int(total_usuarios * prop) for tipo, prop in tipos_clientes.items()}

# --- Lista de CustomerID repetidos según frecuencia ---
customer_orders = []

# Proveedores: muchas compras (50-200 órdenes)
proveedores = df_usuarios.sample(ordenes_por_tipo["Proveedor"])
for cid in proveedores["CustomerID"]:
    customer_orders += [cid] * random.randint(50, 200)

# Frecuentes: compras moderadas (5-15 órdenes)
frecuentes = df_usuarios.drop(proveedores.index).sample(ordenes_por_tipo["Frecuente"])
for cid in frecuentes["CustomerID"]:
    customer_orders += [cid] * random.randint(5, 15)

# Ocasionales: pocas compras (2-4 órdenes)
resto = df_usuarios.drop(proveedores.index).drop(frecuentes.index)
ocasionales = resto.sample(ordenes_por_tipo["Ocasional"])
for cid in ocasionales["CustomerID"]:
    customer_orders += [cid] * random.randint(2, 4)

# Únicos: solo 1 compra
unicos = resto.drop(ocasionales.index)
for cid in unicos["CustomerID"]:
    customer_orders.append(cid)

# --- Ajustar al total exacto de registros del dataset ---
random.shuffle(customer_orders)
if len(customer_orders) > cantidad_ordenes:
    customer_orders = customer_orders[:cantidad_ordenes]
elif len(customer_orders) < cantidad_ordenes:
    faltantes = cantidad_ordenes - len(customer_orders)
    customer_orders += random.choices(df_usuarios["CustomerID"], k=faltantes)

# --- Crear DataFrame temporal para asignar nombres ---
df_customer = pd.DataFrame({"CustomerID": customer_orders})
df_customer = df_customer.merge(df_usuarios, on="CustomerID", how="left")

# --- Sustituir en tu dataset DF ---
DF["CustomerID"] = df_customer["CustomerID"].values
DF["CustomerName"] = df_customer["CustomerName"].values

DF.to_csv("Amazon_simulado.csv", index=False, encoding="utf-8")


# --- Verificación ---
print("Total registros:", len(DF))
print("Usuarios distintos:", DF["CustomerID"].nunique())
print("Primeros registros:\n", DF.head())
print("Distribución de compras por usuario ejemplo:\n", DF["CustomerID"].value_counts().head(10))