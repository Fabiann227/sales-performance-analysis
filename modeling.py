"""Regresi tren sales bulanan per produk (materi: Supervised Learning & Regression).

Langkah mengikuti materi:
  1. Siapkan X (fitur) dan y (target)
  2. Split train 70% / validasi 15% / test 15%
  3. Baseline: Linear Regression  (+ Polynomial, Ridge, Lasso sebagai pembanding)
  4. Metrik: MAE, MSE, RMSE, R2
  5. Pilih konfigurasi pada VALIDASI, evaluasi akhir sekali pada TEST

Target : Sales bulanan per produk (order Completed).
Fitur  : t = urutan bulan (0..29).
Split  : kronologis per bulan, bukan acak, agar data masa depan tidak bocor ke training.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

TEMP = Path(__file__).parent / "temp"

# ---------------------------------------------------------------- 1. data X, y
df = pd.read_csv(TEMP / "clean_merged.csv", parse_dates=["OrderDate"])
df = df[df["Status"] == "Completed"]
df["Month"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()

products = df[["ProductID", "ProductName", "Category"]].drop_duplicates()
all_months = pd.date_range(df["Month"].min(), df["Month"].max(), freq="MS")
grid = products.merge(pd.DataFrame({"Month": all_months}), how="cross")
monthly = (
    grid.merge(df.groupby(["ProductID", "Month"], as_index=False)["Sales"].sum(),
               how="left", on=["ProductID", "Month"])
    .fillna({"Sales": 0.0})  # bulan tanpa penjualan = 0
)
monthly["t"] = monthly["Month"].map({m: i for i, m in enumerate(all_months)})

# ------------------------------------------------------- 2. split 70 / 15 / 15
n = len(all_months)
n_val = round(n * 0.15)
n_train = round(n * 0.70)
train_m = all_months[:n_train]
val_m = all_months[n_train:n_train + n_val]
test_m = all_months[n_train + n_val:]
split = {"train": train_m, "validasi": val_m, "test": test_m}
for name, ms in split.items():
    print(f"{name:9s}: {len(ms):2d} bulan ({ms[0]:%Y-%m} .. {ms[-1]:%Y-%m})")

sets = {name: monthly[monthly["Month"].isin(ms)] for name, ms in split.items()}
for name, d in sets.items():
    d.to_csv(TEMP / f"{name}_monthly.csv", index=False)
    print(f"  {name:9s} {len(d)} baris")

FEATURE, TARGET = ["t"], "Sales"


# ------------------------------------------------------------------ 3. model
def poly(degree):
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False), LinearRegression())


models = {
    "Naive (rata-rata train)": lambda: DummyRegressor(strategy="mean"),
    "Linear (baseline)": lambda: LinearRegression(),
    "Polynomial 2": lambda: poly(2),
    "Polynomial 5": lambda: poly(5),
}
for a in (0.1, 1, 10):
    models[f"Ridge (a={a})"] = lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a))
    models[f"Lasso (a={a})"] = lambda a=a: make_pipeline(StandardScaler(), Lasso(alpha=a, max_iter=10000))


def predict_per_product(make_model, train, others):
    """Satu model per produk (tiap produk punya pola tren sendiri)."""
    preds = {k: pd.Series(index=o.index, dtype=float) for k, o in others.items()}
    for pid, g in train.groupby("ProductID"):
        m = make_model().fit(g[FEATURE], g[TARGET])
        for k, o in others.items():
            go = o[o["ProductID"] == pid]
            preds[k].loc[go.index] = m.predict(go[FEATURE])
    return preds


# --------------------------------------------------------------- 4. metrik
def metrics(y, yhat):
    mse = mean_squared_error(y, yhat)
    return {"MAE": mean_absolute_error(y, yhat), "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y, yhat)}


rows = []
for name, make in models.items():
    p = predict_per_product(make, sets["train"], {k: sets[k] for k in ("train", "validasi", "test")})
    row = {"model": name}
    for k in ("train", "validasi", "test"):
        for mk, mv in metrics(sets[k][TARGET], p[k]).items():
            row[f"{k}_{mk}"] = mv
    rows.append(row)
res = pd.DataFrame(rows)

# ------------------------------------------- 5. pilih di VALIDASI, lapor TEST
show = ["model", "train_RMSE", "validasi_RMSE", "validasi_R2", "test_MAE", "test_MSE", "test_RMSE", "test_R2"]
print("\n=== Perbandingan model (RMSE; makin kecil makin baik) ===")
print(res[show].round(3).to_string(index=False))
candidates = res[res["model"] != "Naive (rata-rata train)"]
best = candidates.loc[candidates["validasi_RMSE"].idxmin(), "model"]
print(f"\nModel terbaik menurut validasi: {best}")
res.round(4).to_csv(TEMP / "model_metrics.csv", index=False)

# --------------------------------------- tren per produk (Linear Regression)
trend = []
for (pid, pname, cat), g in monthly.groupby(["ProductID", "ProductName", "Category"]):
    tr = g[g["Month"].isin(train_m)]
    lr_all = LinearRegression().fit(g[FEATURE], g[TARGET])
    lr_tr = LinearRegression().fit(tr[FEATURE], tr[TARGET])
    trend.append({
        "ProductID": pid, "ProductName": pname, "Category": cat,
        "avg_monthly_sales": g[TARGET].mean(),
        "slope_train": lr_tr.coef_[0], "slope_all": lr_all.coef_[0],
        "R2_all": lr_all.score(g[FEATURE], g[TARGET]),
    })
trend = pd.DataFrame(trend).sort_values("slope_train")
trend["trend_train"] = np.where(trend["slope_train"] < 0, "turun", "naik/stabil")
print("\n=== Slope Linear Regression per produk (per bulan) ===")
print(trend.round(2).to_string(index=False))
trend.round(4).to_csv(TEMP / "product_trend.csv", index=False)
