"""Cleaning + merge dataset raw (customers, orders, payments, products).

Mengikuti materi Cleaning.pdf: cek missing value, strategi per tipe kolom
(hapus baris jika <5% & random, median untuk numerik, "Unknown"/modus untuk
kategorikal), hapus duplikat, dan transformasi tipe data dengan pandas.
Output: temp/clean_merged.csv
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
RAW = BASE / "datasets"
OUT = BASE / "temp"
OUT.mkdir(exist_ok=True)

customers = pd.read_csv(RAW / "customers.csv")
orders = pd.read_csv(RAW / "orders.csv")
payments = pd.read_csv(RAW / "payments.csv")
products = pd.read_csv(RAW / "products.csv")

print("=== RAW: missing value per kolom ===")
for name, df in [("customers", customers), ("orders", orders), ("payments", payments), ("products", products)]:
    miss = df.isnull().sum()
    print(f"{name} {df.shape}: {miss[miss > 0].to_dict()}")

# ---------------------------------------------------------------- customers
# Age (numerik, ~1.8%) -> imputasi median (tidak sensitif outlier)
customers["Age"] = customers["Age"].fillna(customers["Age"].median())
# City: samakan penulisan dulu, baru tangani missing.
# Missing City diisi "Unknown" (bukan modus) agar tidak membiaskan ke Tehran.
customers["City"] = customers["City"].str.strip().str.title().replace({"Mashad": "Mashhad"})
customers["City"] = customers["City"].fillna("Unknown")
customers["Age"] = customers["Age"].astype(int)
customers["SignupDate"] = pd.to_datetime(customers["SignupDate"], format="%Y-%m-%d")

# ------------------------------------------------------------------- orders
n0 = len(orders)
# Duplikat: baris identik penuh (OrderID unik per transaksi) -> drop
orders = orders.drop_duplicates()
assert not orders["OrderID"].duplicated().any()
print(f"\nDuplikat orders dihapus: {n0 - len(orders)}")

# Kolom yang menentukan nominal penjualan / waktu (OrderDate, Quantity, Discount):
# tidak diimputasi (kasus A materi), missing <1% & acak -> dropna.
# Quantity <= 0 tidak valid (pembelian minimal 1 unit) -> diperlakukan sebagai invalid & dihapus.
n1 = len(orders)
orders = orders.dropna(subset=["OrderDate", "Quantity", "Discount"])
print(f"Baris missing OrderDate/Quantity/Discount dihapus: {n1 - len(orders)}")
n2 = len(orders)
orders = orders[orders["Quantity"] > 0]
print(f"Baris Quantity <= 0 (invalid) dihapus: {n2 - len(orders)}")

# PaymentMethod (kategorikal, ~0.9%) -> "Unknown" (tidak ada sumber untuk menebak)
orders["PaymentMethod"] = orders["PaymentMethod"].fillna("Unknown")

orders["OrderDate"] = pd.to_datetime(orders["OrderDate"], format="%Y-%m-%d")
orders["Quantity"] = orders["Quantity"].astype(int)

# ----------------------------------------------------------------- payments
# PaymentDate identik dengan OrderDate (selisih hanya pada baris missing) -> redundan, dibuang
payments = payments[["OrderID", "PaymentStatus"]]

# -------------------------------------------------------------------- merge
# validate memastikan relasi benar (many-to-one / one-to-one), tidak ada fan-out
df = (
    orders
    .merge(payments, on="OrderID", how="left", validate="one_to_one")
    .merge(products, on="ProductID", how="left", validate="many_to_one")
)
# inner join ke customers: CustomerID 999999 (30 baris) tidak ada di master customer -> orphan, dibuang
n3 = len(df)
df = df.merge(customers, on="CustomerID", how="inner", validate="many_to_one")
print(f"Baris orphan CustomerID dihapus: {n3 - len(df)}")

# --------------------------------------------------------------- turunan
df["Sales"] = (df["Quantity"] * df["UnitPrice"] * (1 - df["Discount"])).round(2)

cols = [
    "OrderID", "CustomerID", "OrderDate", "ProductID", "Quantity", "Discount",
    "PaymentMethod", "Status", "PaymentStatus",
    "Age", "City", "SignupDate", "CustomerSegment",
    "ProductName", "Category", "UnitPrice", "Sales",
]
df = df[cols].sort_values("OrderID").reset_index(drop=True)

# ------------------------------------------------------------ validasi akhir
assert df.isnull().sum().sum() == 0, "masih ada missing value"
assert not df.duplicated().any() and not df["OrderID"].duplicated().any()
print("\n=== HASIL ===")
print(df.shape)
df.info()
print("Order sebelum SignupDate (hanya dicatat, tidak dihapus):", (df["OrderDate"] < df["SignupDate"]).sum())

df.to_csv(OUT / "clean_merged.csv", index=False)
print("Tersimpan:", OUT / "clean_merged.csv")
