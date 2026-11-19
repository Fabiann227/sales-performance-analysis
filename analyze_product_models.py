import pandas as pd

# Load per-product walk-forward evaluation results.
df = pd.read_csv("temp/walk_forward_product_metrics.csv")

# Validate the input data.
required_columns = {
    "ProductID",
    "ProductName",
    "fold",
    "model",
    "RMSE",
    "WAPE",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

if df.empty:
    raise ValueError("The evaluation dataset is empty.")

# 1. Calculate average performance across all folds.
summary = (
    df.groupby(["ProductID", "ProductName", "model"])
    .agg(
        mean_RMSE=("RMSE", "mean"),
        mean_WAPE=("WAPE", "mean"),
        folds=("fold", "nunique"),
    )
    .reset_index()
)

# 2. Identify the winning model in each product/fold combination.
fold_winners = (
    df.sort_values(
        ["RMSE", "model"],
        ascending=[True, True],
    )
    .groupby(["ProductID", "fold"], as_index=False)
    .first()
)

# 3. Count how many folds each model won.
wins = (
    fold_winners.groupby(["ProductID", "model"])
    .size()
    .rename("fold_wins")
    .reset_index()
)

# 4. Combine average performance and fold-win counts.
summary = summary.merge(
    wins,
    on=["ProductID", "model"],
    how="left",
)

summary["fold_wins"] = summary["fold_wins"].fillna(0).astype(int)

# 5. Rank models independently for each product.
summary = summary.sort_values(["ProductID", "mean_RMSE", "model"])

summary["rank"] = (
    summary.groupby("ProductID")["mean_RMSE"].rank(method="first").astype(int)
)

# 6. Select the top two models per product.
top_two = summary[summary["rank"] <= 2].copy()

# Display results.
columns = [
    "ProductID",
    "ProductName",
    "model",
    "mean_RMSE",
    "mean_WAPE",
    "fold_wins",
    "folds",
    "rank",
]

print("\n=== TOP TWO MODELS PER PRODUCT ===\n")
print(top_two[columns].round(2).to_string(index=False))

# 7. Save the complete summary and top-two results.
summary.to_csv(
    "temp/product_model_summary.csv",
    index=False,
)

top_two.to_csv(
    "temp/product_model_top_two.csv",
    index=False,
)

print("\nAnalysis completed.")
print("Saved:")
print("- temp/product_model_summary.csv")
print("- temp/product_model_top_two.csv")
