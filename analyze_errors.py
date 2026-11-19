import pandas as pd
import numpy as np

INPUT_FILE = "temp/final_test_predictions.csv"
OUTPUT_FILE = "temp/final_test_error_analysis.csv"

df = pd.read_csv(INPUT_FILE)

# 1. Calculate prediction errors.
df["Error"] = df["ActualSales"] - df["PredictedSales"]
df["AbsoluteError"] = df["Error"].abs()
df["SquaredError"] = df["Error"] ** 2

# 2. Aggregate errors by strategy and product.
product_errors = (
    df.groupby(["strategy", "ProductID", "ProductName", "Category"])
    .agg(
        TotalAbsoluteError=("AbsoluteError", "sum"),
        MeanAbsoluteError=("AbsoluteError", "mean"),
        RMSE=("SquaredError", lambda x: np.sqrt(x.mean())),
        ActualSales=("ActualSales", "sum"),
    )
    .reset_index()
)

# 3. Calculate each product's contribution to total error.
product_errors["ErrorContribution"] = (
    product_errors["TotalAbsoluteError"]
    / product_errors.groupby("strategy")["TotalAbsoluteError"].transform("sum")
    * 100
)

# 4. Display the five largest error contributors.
for strategy in product_errors["strategy"].unique():
    print(f"\n=== TOP 5 ERROR CONTRIBUTORS: {strategy} ===\n")

    result = (
        product_errors.loc[product_errors["strategy"] == strategy]
        .sort_values("TotalAbsoluteError", ascending=False)
        .head(5)
    )

    print(
        result[
            [
                "ProductID",
                "ProductName",
                "TotalAbsoluteError",
                "MeanAbsoluteError",
                "RMSE",
                "ErrorContribution",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

# 5. Compare total error contribution across strategies.
print("\n=== OVERALL ERROR SUMMARY ===\n")

summary = (
    df.groupby("strategy")
    .agg(
        TotalAbsoluteError=("AbsoluteError", "sum"),
        MAE=("AbsoluteError", "mean"),
        RMSE=("SquaredError", lambda x: np.sqrt(x.mean())),
        TotalActualSales=("ActualSales", "sum"),
    )
    .reset_index()
)

summary["WAPE"] = summary["TotalAbsoluteError"] / summary["TotalActualSales"] * 100

print(summary.round(2).to_string(index=False))

# 6. Save results.
product_errors.to_csv(OUTPUT_FILE, index=False)

print(f"\nAnalysis completed.")
print(f"Saved: {OUTPUT_FILE}")
f
