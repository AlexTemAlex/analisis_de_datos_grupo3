import pandas as pd
import random


precios_ecommerce = {
# ===================== ELECTRONICS =====================
"Smartphone X Pro": 999,
"Smartphone X Lite": 499,
"Smartphone X Max": 1099,
"Wireless Earbuds Pro": 199,
"Wireless Earbuds Mini": 79,
"Noise Cancelling Headphones": 299,
"Bluetooth Speaker Mini": 49,
"Bluetooth Speaker XL": 179,
"4K Smart TV 55in": 599,
"4K Smart TV 65in": 899,
"Laptop Ultrabook 14in": 1099,
"Laptop Ultrabook 16in": 1399,
"Gaming Laptop RTX": 1899,
"Tablet Pro 11in": 799,
"Tablet Mini 8in": 399,
"Smartwatch Series 5": 349,
"Smartwatch Sport Edition": 279,
"Fitness Band Lite": 59,
"Fitness Band Pro": 129,
"Mechanical Keyboard RGB": 149,
"Mechanical Keyboard Low Profile": 129,
"Gaming Mouse Pro": 99,
"Gaming Mouse Wireless": 129,
"External SSD 1TB": 119,
"External SSD 2TB": 219,
"External HDD 2TB": 89,
"USB-C Charger 65W": 39,
"Wireless Charger Pad": 29,
"WiFi Router AX3000": 159,
"WiFi Mesh System": 299,
"Action Camera 4K": 249,
"Drone Mini": 399,
"Drone Pro Camera": 999,
"Webcam Full HD": 69,
"Webcam 4K": 149,
"Monitor 27in QHD": 329,
"Monitor 32in 4K": 599,
"Graphics Tablet Pro": 349,
"Microphone USB": 129,
"HDMI Cable 2m": 15,
"Power Bank 20000mAh": 49,
# ===================== BOOKS =====================
"Novel Bestseller Hardcover": 29,
"Novel Bestseller Paperback": 18,
"Children's Picture Book": 15,
"Children's Learning Book": 17,
"Science Fiction Novel": 20,
"Fantasy Saga Book 1": 22,
"Fantasy Saga Book 2": 22,
"Self Help Motivation Book": 19,
"Business Strategy Book": 25,
"Python Programming Guide": 39,
"Data Science Handbook": 45,
"Machine Learning Basics": 42,
"Cooking Recipes Book": 30,
"Healthy Lifestyle Guide": 24,
"Travel Photography Book": 35,
"History of Technology": 38,
# ==================== HOME & KITCHEN =====================
"Air Fryer XL": 179,
"Air Fryer Compact": 119,
"Instant Pot Pro": 199,
"Electric Kettle Stainless": 59,
"Cookware Set 10pcs": 249,
"Cookware Set Non-Stick": 179,
"Vacuum Cleaner Robot": 499,
"Vacuum Cleaner Bagless": 229,
"LED Desk Lamp": 49,
"Smart Light Bulb": 29,
"Desk Organizer Set": 35,
"Water Bottle Insulated": 29,
"Coffee Maker Drip": 89,
"Coffee Maker Espresso": 299,
"Blender High Power": 149,
"Rice Cooker Digital": 129,
"Office Chair Ergonomic": 349,
"Standing Desk Converter": 279,
# ===================== CLOTHING =====================
"T-Shirt Cotton Basic": 15,
"T-Shirt Sport DryFit": 25,
"Jeans Slim Fit": 49,
"Jeans Regular Fit": 45,
"Dress Shirt Formal": 39,
"Dress Shirt Casual": 35,
"Winter Jacket Waterproof": 149,
"Hoodie Pullover": 59,
"Running Shorts": 29,
"Athletic Leggings": 39,
"Sneakers Casual": 69,
"Running Shoes Pro": 129,
"Baseball Cap": 19,
"Sunglasses UV Protection": 49,
"Backpack Travel": 89,
"Backpack Laptop": 79,
# ===================== TOYS & GAMES =====================
"Board Game Family Edition": 39,
"Board Game Strategy": 59,
"Puzzle 1000 Pieces": 25,
"Puzzle 500 Pieces": 18,
"Kids Toy Car": 29,
"Remote Control Car": 79,
"Building Blocks Set": 49,
"Educational Toy Math": 35,
"Action Figure Superhero": 29,
"Doll Fashion Set": 45,
"Card Game Party": 19,
"Chess Wooden Set": 59,
# ===================== SPORTS & OUTDOORS =====================
"Yoga Mat Non-Slip": 29,
"Yoga Mat Extra Thick": 45,
"Fitness Resistance Bands": 35,
"Dumbbell Set Adjustable": 249,
"Treadmill Foldable": 899,
"Exercise Bike Indoor": 699,
"Camping Tent 4-Person": 179,
"Sleeping Bag Thermal": 89,
"Hiking Backpack 40L": 129,
"Water Bottle Sport": 19,
"Fishing Rod Combo": 99,
"Basketball Outdoor": 29,
# ===================== BEAUTY & PERSONAL CARE =====================
"Electric Toothbrush": 79,
"Hair Dryer Ionic": 69,
"Skincare Gift Set": 59,
"Face Cleansing Brush": 49,
"Men Grooming Kit": 65,
"Makeup Brush Set": 39,
# ===================== HEALTH & HOUSEHOLD =====================
"First Aid Kit": 29,
"Digital Thermometer": 15,
"Blood Pressure Monitor": 59,
"Air Purifier HEPA": 249,
"Disinfectant Wipes Pack": 12,
"Laundry Detergent Eco": 18,
# ===================== GROCERY & GOURMET =====================
"Organic Coffee Beans": 18,
"Green Tea Premium": 15,
"Protein Powder Vanilla": 49,
"Protein Powder Chocolate": 49,
"Olive Oil Extra Virgin": 22,
"Honey Natural Raw": 19,
# ===================== PET SUPPLIES =====================
"Dog Food Premium": 55,
"Cat Food Indoor": 45,
"Pet Bed Comfort": 69,
"Pet Travel Carrier": 89,
"Dog Leash Reflective": 19,
"Cat Scratching Post": 79,
# ===================== AUTOMOTIVE =====================
"Car Phone Mount": 25,
"Car Vacuum Cleaner": 79,
"Jump Starter Power Pack": 129,
"Car Seat Organizer": 29,
"Tire Pressure Gauge": 15,
"Windshield Sun Shade": 25,
# ===================== BABY =====================
"Baby Monitor Camera": 159,
"Baby Bottle Set": 35,
"Diapers Size 3 Pack": 49,
"Baby Stroller Lightweight": 299,
"Baby Play Mat": 69,
"High Chair Adjustable": 189,
# ===================== TOOLS & HOME IMPROVEMENT =====================
"Cordless Drill Set": 149,
"Tool Kit 100pcs": 99,
"Smart Thermostat": 249,
"LED Flashlight Rechargeable": 39,
"Measuring Laser Tool": 89,
"Extension Cord Heavy Duty": 29,
# ===================== OFFICE PRODUCTS =====================
"Notebook A4 Pack": 12,
"Office Paper Ream": 9,
"Wireless Mouse Office": 29,
"Desk Calendar": 15,
"Pen Set Premium": 25,
"Document Organizer": 35,
# ===================== GARDEN & OUTDOOR =====================
"Garden Hose Expandable": 39,
"Plant Watering Can": 19,
"Outdoor Solar Lights": 45,
"BBQ Grill Portable": 179,
"Lawn Sprinkler System": 89,
"Garden Tool Set": 79,
# ===================== FURNITURE =====================
"Office Desk Wood": 349,
"Bookshelf 5-Tier": 199,
"Coffee Table Modern": 249,
"Sofa 3-Seater": 899,
"Bed Frame Queen": 699,
"Nightstand Drawer": 129
}

# -------------------------
# 1. DataFrame de productos
# -------------------------
df_productos = (
    pd.DataFrame.from_dict(precios_ecommerce, orient="index", columns=["UnitPrice"])
    .reset_index()
    .rename(columns={"index": "ProductName"})
)

# -------------------------
# 2. Asignar categoría por nombre
# -------------------------
def asignar_categoria(nombre):
    for categoria in marcas_ecommerce.keys():
        if categoria.lower() in nombre.lower():
            return categoria
    return None

# Asignación manual correcta (robusta)
categorias_map = {
    "Electronics": [
        "Smartphone", "Earbuds", "Headphones", "Speaker", "TV", "Laptop",
        "Tablet", "Smartwatch", "Fitness Band", "Keyboard", "Mouse",
        "SSD", "HDD", "Charger", "Router", "Camera", "Drone",
        "Webcam", "Monitor", "Tablet", "Microphone", "Power Bank"
    ],
    "Books": ["Book", "Novel", "Guide", "Handbook"],
    "Home & Kitchen": ["Air Fryer", "Cookware", "Vacuum", "Lamp", "Coffee", "Blender", "Rice", "Chair", "Desk"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Hoodie", "Shorts", "Leggings", "Sneakers", "Shoes", "Cap", "Sunglasses", "Backpack"],
    "Toys & Games": ["Game", "Puzzle", "Toy", "Blocks", "Figure", "Doll", "Chess"],
    "Sports & Outdoors": ["Yoga", "Dumbbell", "Treadmill", "Bike", "Camping", "Sleeping", "Hiking", "Fishing", "Basketball"],
    "Beauty & Personal Care": ["Toothbrush", "Hair", "Skincare", "Makeup", "Grooming"],
    "Health & Household": ["First Aid", "Thermometer", "Pressure", "Purifier", "Detergent"],
    "Grocery & Gourmet Food": ["Coffee", "Tea", "Protein", "Oil", "Honey"],
    "Pet Supplies": ["Dog", "Cat", "Pet"],
    "Automotive": ["Car", "Tire", "Windshield"],
    "Baby": ["Baby", "Diapers", "Stroller", "High Chair"],
    "Tools & Home Improvement": ["Drill", "Tool", "Thermostat", "Flashlight", "Laser", "Extension"],
    "Office Products": ["Notebook", "Paper", "Mouse", "Calendar", "Pen", "Organizer"],
    "Garden & Outdoor": ["Garden", "Plant", "BBQ", "Lawn"],
    "Furniture": ["Desk", "Bookshelf", "Table", "Sofa", "Bed", "Nightstand"]
}

def categoria_por_nombre(nombre):
    for cat, keywords in categorias_map.items():
        if any(k.lower() in nombre.lower() for k in keywords):
            return cat
    return "Electronics"

df_productos["Category"] = df_productos["ProductName"].apply(categoria_por_nombre)

# -------------------------
# 3. Asignar marca coherente
# -------------------------
marcas_ecommerce = {
    'Books': ['ReadMore', 'StoryTime', 'PageTurner', 'NovelWorld'],
    'Home & Kitchen': ['HomeLux', 'CookEase', 'ZenSpace', 'KitchenPro'],
    'Clothing': ['UrbanStyle', 'FitLife', 'TrendWear', 'ComfortCo'],
    'Toys & Games': ['FunZone', 'PlayTime', 'ToyMaster', 'JoyLand'],
    'Sports & Outdoors': ['ActivePro', 'Sporty', 'FitGear', 'OutdoorX'],
    'Electronics': ['Apex', 'BrightLux', 'CoreTech', 'NexPro'],
    'Beauty & Personal Care': ['GlowWell', 'BeautyEssence', 'CarePlus', 'FreshLook'],
    'Health & Household': ['HealthPro', 'SafeHome', 'WellBeing', 'CleanEase'],
    'Grocery & Gourmet Food': ['Foodies', 'GourmetX', 'TasteBest', 'FreshHarvest'],
    'Pet Supplies': ['PetLovers', 'FurryFriend', 'PawPal', 'HappyPets'],
    'Automotive': ['AutoPro', 'DriveMax', 'CarEase', 'MotorPlus'],
    'Baby': ['BabyJoy', 'TinySteps', 'LittleCare', 'CuddleTime'],
    'Tools & Home Improvement': ['ToolMaster', 'FixIt', 'BuildPro', 'HandyGear'],
    'Office Products': ['OfficePro', 'DeskMate', 'WorkSmart', 'PaperPlus'],
    'Garden & Outdoor': ['GreenThumb', 'GardenPro', 'PlantEase', 'OutdoorLife'],
    'Furniture': ['ComfyHome', 'FurniLux', 'RoomStyle', 'HomeSpace']
}


df_productos["Brand"] = df_productos["Category"].apply(
    lambda c: random.choice(marcas_ecommerce[c])
)

# -------------------------
# 4. Crear ProductID
# -------------------------
df_productos["ProductID"] = [
    f"PROD{str(i).zfill(4)}" for i in range(1, len(df_productos) + 1)
]







# -------------------------
# Cargar dataset
# -------------------------
DF = pd.read_csv("Amazon_simulado.csv")

# -------------------------
# Asignar productos aleatoriamente
# -------------------------
productos_sample = df_productos.sample(
    n=len(DF), replace=True, random_state=42
).reset_index(drop=True)

DF["ProductID"] = productos_sample["ProductID"]
DF["ProductName"] = productos_sample["ProductName"]
DF["Category"] = productos_sample["Category"]
DF["Brand"] = productos_sample["Brand"]
DF["UnitPrice"] = productos_sample["UnitPrice"]


DF["TotalAmount"] = DF["Quantity"] * DF["UnitPrice"]


DF.to_csv("amazon_actualizado.csv", index=False)


print(DF.nunique()[["ProductID", "ProductName", "Category", "Brand"]])
