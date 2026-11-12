import pandas as pd

forecast = pd.read_csv("temp/future_forecast.csv")

# 1. Check total predictions
print("Total predictions:", len(forecast))

# 2. Check unique products and months
print("Unique products:", forecast["ProductID"].nunique())
print("Forecast months:", forecast["Month"].nunique())

# 3. Check selected model distribution
print("\nModel distribution:")
print(forecast.groupby("model")["ProductID"].nunique())

# 4. Check negative or missing predictions
print("\nNegative predictions:")
print((forecast["PredictedSales"] < 0).sum())

print("\nMissing values:")
print(forecast.isna().sum())

# 5. Check duplicate product-month combinations
print("\nDuplicate product-month combinations:")
print(forecast.duplicated(["ProductID", "Month"]).sum())

# 6. Show all forecasts
print("\nForecast results:")
print(forecast.sort_values(["ProductID", "Month"]).to_string(index=False))
