import pandas as pd

DF = pd.read_csv("amazon_actualizado2.csv")

# Eliminar la columna TipoCliente si existe
DF = DF.drop(columns=["TipoCliente"], errors="ignore")

# Recalcular SOLO TotalAmount
DF["TotalAmount"] = (
    DF["Quantity"] * DF["UnitPrice"]
    - DF.get("Discount", 0)
    + DF.get("Tax", 0)
    + DF.get("ShippingCost", 0)
)

# Guardar archivo final
DF.to_csv("amazon_actualizado3.csv", index=False)
