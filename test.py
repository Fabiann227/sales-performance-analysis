import pandas as pd

df = pd.read_csv("temp/final_test_product_metrics.csv")

# Compare per-product performance across strategies.
comparison = df.pivot(
    index=["ProductID", "ProductName"],
    columns="strategy",
    values="RMSE",
).reset_index()

# Calculate which strategy performs best for each product.
strategies = [
    "Global Training Mean Baseline",
    "Best Global ML",
    "Per-Product Selection",
]

comparison["Best Strategy"] = comparison[strategies].idxmin(axis=1)

comparison = comparison.sort_values("ProductID")

print("\n=== FINAL TEST RMSE BY PRODUCT ===\n")
print(comparison.round(2).to_string(index=False))

print("\n=== STRATEGY WIN COUNTS ===\n")
print(comparison["Best Strategy"].value_counts())

comparison.to_csv(
    "temp/final_test_strategy_comparison.csv",
    index=False,
)

print("\nSaved: temp/final_test_strategy_comparison.csv")
