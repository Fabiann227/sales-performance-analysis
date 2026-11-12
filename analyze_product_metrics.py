import pandas as pd

NAIVE = "Naive (rata-rata train)"

df = pd.read_csv("temp/product_model_metrics.csv")

# Best overall model per product, selected using validation RMSE.
best_overall = df.sort_values("validasi_RMSE").drop_duplicates(subset="ProductID")

# Best ML model per product, excluding Naive.
best_ml = (
    df.loc[df["model"] != NAIVE]
    .sort_values("validasi_RMSE")
    .drop_duplicates(subset="ProductID")
)

print("\n=== Best overall model per product ===")
print(
    best_overall[
        ["ProductID", "ProductName", "model", "validasi_RMSE", "test_RMSE"]
    ].to_string(index=False)
)

print("\n=== Best ML model per product ===")
print(
    best_ml[
        ["ProductID", "ProductName", "model", "validasi_RMSE", "test_RMSE"]
    ].to_string(index=False)
)

# Compare Naive against the best ML model for each product.
naive = df.loc[
    df["model"] == NAIVE,
    ["ProductID", "validasi_RMSE", "test_RMSE"],
].rename(
    columns={
        "validasi_RMSE": "naive_validasi_RMSE",
        "test_RMSE": "naive_test_RMSE",
    }
)

comparison = best_ml.merge(
    naive,
    on="ProductID",
)

comparison["validation_improvement"] = (
    comparison["naive_validasi_RMSE"] - comparison["validasi_RMSE"]
)

print("\n=== Best ML vs Naive ===")
print(
    comparison[
        [
            "ProductName",
            "model",
            "naive_validasi_RMSE",
            "validasi_RMSE",
            "validation_improvement",
        ]
    ].to_string(index=False)
)
