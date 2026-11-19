import pandas as pd

df = pd.read_csv("temp/final_test_predictions.csv")

print("\n=== DATASET COLUMNS ===")
print(df.columns.tolist())

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== SAMPLE DATA ===")
print(df.head(10).to_string(index=False))

print("\n=== DATA TYPES ===")
print(df.dtypes)
