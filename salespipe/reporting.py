"""Local, self-contained HTML report generation."""

from __future__ import annotations

import html
import shutil
from pathlib import Path
from typing import Any, cast

import pandas as pd


def _table(df: pd.DataFrame, digits: int = 2) -> str:
    display = df.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].round(digits)
    return display.to_html(index=False, classes="data-table", border=0, na_rep="—")


def _list(items: list[str]) -> str:
    if not items:
        return "<p>None identified from the available artifacts.</p>"
    return "<ul>" + "".join(
        f"<li>{html.escape(item)}</li>" for item in items
    ) + "</ul>"


def _period(df: pd.DataFrame) -> str:
    dates = pd.to_datetime(df["Month"])
    return f"{dates.min():%B %Y} to {dates.max():%B %Y}"


def _forecast_table(forecast_products: pd.DataFrame, sort_column: str, ascending: bool) -> str:
    columns = [
        "ProductName",
        "model",
        "ForecastTotal",
        "FirstForecastSales",
        "LastForecastSales",
        "ForecastChange",
    ]
    selected = cast(
        pd.DataFrame,
        forecast_products.sort_values(by=sort_column, ascending=ascending).head(5)[columns],
    )
    return _table(selected)


def generate_report(
    temp_dir: Path,
    visualization_dir: Path,
    report_dir: Path,
    insights: dict[str, Any],
) -> Path:
    """Build the analytical report and copy chart assets into its local assets folder."""
    assets = report_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    chart_names = [
        "overall_monthly_sales.png",
        "product_monthly_sales.png",
        "product_sales_comparison.png",
        "model_performance_comparison.png",
        "actual_vs_predicted_sales.png",
        "forecasting_error_contribution.png",
        "overall_future_forecast.png",
        "future_forecast_by_product.png",
    ]
    for name in chart_names:
        source = visualization_dir / name
        if source.exists():
            shutil.copy2(source, assets / name)

    metrics = pd.read_csv(temp_dir / "final_test_metrics.csv")
    walk_forward = pd.read_csv(temp_dir / "walk_forward_summary.csv")
    trends = pd.read_csv(temp_dir / "product_trend.csv")
    monthly = pd.read_csv(temp_dir / "monthly.csv")
    clean = pd.read_csv(temp_dir / "clean_merged.csv")
    forecast = pd.read_csv(temp_dir / "future_forecast.csv")
    train = pd.read_csv(temp_dir / "train.csv")
    validation = pd.read_csv(temp_dir / "validasi.csv")
    test = pd.read_csv(temp_dir / "test.csv")
    error_details = insights["error_details"]
    forecast_products = insights["forecast_products"]

    best_strategy = str(insights["best_strategy"])
    best_walk_forward = str(insights["best_walk_forward_model"])
    best_metrics = metrics.loc[metrics["strategy"] == best_strategy].iloc[0]
    forecast["Month"] = pd.to_datetime(forecast["Month"])
    forecast_months = cast(pd.Series, forecast["Month"]).sort_values(ascending=True).drop_duplicates()
    historical_months = cast(
        pd.Series,
        pd.to_datetime(monthly["Month"]),
    ).sort_values(ascending=True).drop_duplicates()

    development_months = historical_months.iloc[:-3].reset_index(drop=True)
    fold_rows = []
    for fold_number, train_end in enumerate((12, 15, 18), start=1):
        fold_rows.append(
            {
                "Fold": fold_number,
                "Training period": f"{development_months.iloc[0]:%b %Y} – {development_months.iloc[train_end - 1]:%b %Y}",
                "Validation period": f"{development_months.iloc[train_end]:%b %Y} – {development_months.iloc[train_end + 2]:%b %Y}",
            }
        )
    fold_table = pd.DataFrame(fold_rows)

    kpis = {
        "Cleaned transactions": f"{len(clean):,}",
        "Products": f"{monthly['ProductID'].nunique():,}",
        "Historical period": f"{historical_months.iloc[0]:%b %Y} – {historical_months.iloc[-1]:%b %Y}",
        "Final-test WAPE": f"{best_metrics['WAPE']:.2f}%",
    }
    cards = "".join(
        f'<div class="card"><strong>{html.escape(value)}</strong><span>{html.escape(label)}</span></div>'
        for label, value in kpis.items()
    )

    def chart(name: str, title: str) -> str:
        return (
            f'<figure><img src="assets/{name}" alt="{html.escape(title)}">'
            f"<figcaption>{html.escape(title)}</figcaption></figure>"
        )

    relative_error = error_details.loc[
        error_details["strategy"] == best_strategy
    ].sort_values("RelativeAbsoluteErrorPct", ascending=False)
    relative_error_table = _table(
        relative_error.head(10)[
            [
                "ProductName",
                "TotalAbsoluteError",
                "MeanAbsoluteError",
                "RMSE",
                "ErrorContribution",
                "RelativeAbsoluteErrorPct",
            ]
        ]
    )
    selected_models_table = _table(
        forecast_products[
            ["ProductName", "model", "ForecastTotal", "ForecastChange"]
        ].sort_values("ProductName")
    )
    trend_table = _table(
        cast(pd.DataFrame, trends.sort_values(by="avg_monthly_sales", ascending=False).head(10)[
            [
                "ProductName",
                "avg_monthly_sales",
                "slope_train",
                "R2_all",
                "trend_train",
            ]
        ])
    )
    recommendations = [
        f"Use {best_strategy} as the aggregate reference strategy for the evaluated final-test horizon.",
        "Treat positive and negative fitted slopes as descriptive summaries rather than statistically established trends.",
        "Review products with high relative error separately from products with high absolute error before using forecasts for planning.",
        "Refresh the analysis when new completed transactions are available and re-evaluate the strategy chronologically.",
    ]

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sales Performance Analysis</title><style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;line-height:1.55;color:#172033;background:#f4f7fb;margin:0}}main{{max-width:1180px;margin:auto;padding:36px}}header{{background:#12233f;color:white;padding:48px 36px;margin:-36px -36px 30px}}h1{{margin:0 0 8px;font-size:2.4rem}}h2{{margin-top:42px;border-bottom:2px solid #dce5f2;padding-bottom:8px}}h3{{color:#245b96}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}}.card{{background:white;border-radius:10px;padding:20px;box-shadow:0 2px 10px #17203312}}.card strong{{display:block;font-size:1.55rem;color:#1769aa}}.card span{{color:#52657e}}.data-table{{border-collapse:collapse;width:100%;background:white;font-size:.88rem}}.data-table th,.data-table td{{padding:8px 10px;border-bottom:1px solid #e4eaf2;text-align:left;white-space:nowrap}}.data-table th{{background:#eaf1f8}}figure{{background:white;padding:14px;border-radius:10px;margin:18px 0;box-shadow:0 2px 10px #17203312}}figure img{{max-width:100%;height:auto;display:block;margin:auto}}figcaption{{text-align:center;color:#52657e;margin-top:8px}}.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px}}.note{{border-left:4px solid #2b79b9;background:#eaf4fb;padding:12px 16px}}.definition{{background:#fffaf0;border-left:4px solid #e0a52b;padding:12px 16px}}footer{{color:#65758a;border-top:1px solid #dce5f2;margin-top:48px;padding-top:18px}}
@media print{{@page{{size:A4;margin:16mm 13mm}}body{{background:white;color:#111;font-size:10pt}}main{{max-width:none;padding:0}}header{{margin:0 0 18px;padding:24px;background:#12233f!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}}h1{{font-size:22pt}}h2{{font-size:15pt;margin-top:26px;break-after:avoid;page-break-after:avoid}}h3{{break-after:avoid;page-break-after:avoid}}.cards{{grid-template-columns:repeat(4,1fr);gap:8px}}.card{{box-shadow:none;border:1px solid #d6dee8;padding:10px;break-inside:avoid;page-break-inside:avoid}}.card strong{{font-size:13pt}}.chart-grid{{display:block}}figure{{box-shadow:none;border:1px solid #d6dee8;break-inside:avoid;page-break-inside:avoid;margin:12px 0 18px;padding:8px}}figure img{{max-height:210mm;width:auto;max-width:100%}}figcaption{{font-size:9pt}}.data-table{{font-size:8pt;table-layout:auto;break-inside:auto}}.data-table thead{{display:table-header-group}}.data-table tr{{break-inside:avoid;page-break-inside:avoid}}.data-table th,.data-table td{{white-space:normal;padding:5px 6px}}section{{break-inside:auto}}.note,.definition{{break-inside:avoid;page-break-inside:avoid}}footer{{font-size:8pt}}}}
</style></head><body><main><header><h1>Sales Performance Analysis</h1><p>CRISP-DM analytical report · Generated from reproducible local pipeline artifacts</p></header>
<section><h2>1. Executive Summary</h2><div class="cards">{cards}</div><p class="note"><b>Measured result:</b> {html.escape(best_strategy)} achieved the lowest final-test RMSE. The strongest walk-forward model was <b>{html.escape(best_walk_forward)}</b>. The final-test WAPE for the selected strategy was <b>{best_metrics['WAPE']:.2f}%</b>.</p><p>The cleaned dataset contains {len(clean):,} transactions across {monthly['ProductID'].nunique()} products from {historical_months.iloc[0]:%B %Y} through {historical_months.iloc[-1]:%B %Y}. Future estimates cover {forecast_months.iloc[0]:%B %Y} through {forecast_months.iloc[-1]:%B %Y}. Forecast changes describe movement between forecast months and are not observed historical growth.</p></section>
<section><h2>2. Business Understanding</h2><p>The project evaluates marketplace sales performance and compares chronological forecasting strategies to support evidence-based planning. The analytical questions are how sales evolve over time, which products contribute most to performance and error, and which forecasting approach performs best on an untouched future period.</p></section>
<section><h2>3. Data Understanding</h2><p>The integrated data contains {len(clean):,} cleaned transactions from {_period(monthly)} and {monthly['ProductID'].nunique()} products. The modeling table contains {len(monthly):,} product-month observations; zero-sales product-month combinations are retained.</p><h3>Historical sales</h3><div class="chart-grid">{chart('overall_monthly_sales.png','Overall monthly sales')}{chart('product_sales_comparison.png','Total sales by product')}{chart('product_monthly_sales.png','Monthly sales trends by product')}</div></section>
<section><h2>4. Data Preparation</h2><p>Raw tables were validated, duplicate orders removed, dates and numeric fields coerced, invalid quantity/discount rows excluded, customer ages imputed with the median, missing categorical values filled with <code>Unknown</code>, tables relationally merged, and transaction sales calculated as Quantity × UnitPrice × (1 − Discount).</p></section>
<section><h2>5. Modeling Methodology</h2><p><b>Chronological split:</b> training covers {_period(train)}, validation covers {_period(validation)}, and testing covers {_period(test)}. Chronological splitting is required because a random split can place future observations in the training data, producing optimistic estimates and temporal leakage. Models are selected using validation RMSE; the final test period remains separate and is used only for the final comparison.</p><h3>Expanding-window walk-forward validation</h3>{_table(fold_table)}<p>Each fold expands the training window while moving the validation window forward. This tests whether model performance is stable across multiple historical origins instead of relying on one validation window.</p></section>
<section><h2>6. Model Evaluation</h2><p>Lower MAE, RMSE, and WAPE are better; higher R² is better. Walk-forward metrics measure repeated validation performance used for model selection. Final-test metrics measure one held-out period after selection. Global strategies apply one model across products, whereas per-product selection allows each product to use its own walk-forward winner.</p>{_table(metrics.sort_values('RMSE'))}<div class="definition"><b>Observed result:</b> The Training Mean Baseline outperformed the evaluated regression models on aggregate final-test metrics and also led the walk-forward summary. This is a result of the observed data and evaluation design; the artifacts do not establish a causal reason why a constant training-period mean was more accurate.</div><div class="chart-grid">{chart('model_performance_comparison.png','Walk-forward model performance')}{chart('actual_vs_predicted_sales.png','Actual versus predicted sales')}</div></section>
<section><h2>7. Sales Performance Analysis</h2><p>Product slopes are fitted linear summaries over the training period. A positive or negative estimated slope describes the direction of the fitted line only. It is not evidence of a statistically significant trend. The low product-level R² values indicate that the linear time feature explains only a small portion of observed sales variation.</p>{trend_table}</section>
<section><h2>8. Future Forecasting</h2><p>The pipeline generated {len(forecast):,} forecasts for {forecast['ProductID'].nunique()} products from {forecast_months.iloc[0]:%B %Y} through {forecast_months.iloc[-1]:%B %Y}, using historical data ending {historical_months.iloc[-1]:%B %Y}. The selected product models below are the walk-forward-selected models used to generate the forecast artifact.</p><h3>Selected model and forecast summary by product</h3>{selected_models_table}<h3>Highest forecasted sales</h3>{_forecast_table(forecast_products, 'ForecastTotal', False)}<h3>Lowest forecasted sales</h3>{_forecast_table(forecast_products, 'ForecastTotal', True)}<h3>Largest forecast-month increases</h3>{_forecast_table(forecast_products, 'ForecastChange', False)}<h3>Largest forecast-month decreases</h3>{_forecast_table(forecast_products, 'ForecastChange', True)}<div class="chart-grid">{chart('overall_future_forecast.png','Overall future forecast')}{chart('future_forecast_by_product.png','Future forecast by product')}</div></section>
<section><h2>9. Forecasting Error Analysis</h2><p>For the best final-test strategy, total absolute error is the sum of absolute errors for a product, MAE is the average absolute error per test observation, and RMSE penalizes larger errors more strongly. Error contribution is a product's share of the strategy's total absolute error. Relative absolute error is defined as total absolute error divided by total actual sales, expressed as a percentage; it is unavailable when actual sales are zero.</p>{relative_error_table}<p>Absolute error contribution is not the same as relative accuracy. A high-volume product can contribute substantial absolute error while having a moderate relative error. Therefore, the largest contributors should not automatically be classified as the least predictable products.</p><div class="chart-grid">{chart('forecasting_error_contribution.png','Forecasting error contribution')}</div></section>
<section><h2>10. Business Insights and Recommendations</h2><h3>Measured findings</h3><ul><li>Best walk-forward model: {html.escape(best_walk_forward)}.</li><li>Best final-test strategy: {html.escape(best_strategy)}, with WAPE of {best_metrics['WAPE']:.2f}%.</li><li>Products with positive estimated training slopes: {html.escape(', '.join(insights['increasing_products']) or 'none')}.</li><li>Products with negative estimated training slopes: {html.escape(', '.join(insights['decreasing_products']) or 'none')}.</li><li>Largest absolute-error contributors under the best strategy: {html.escape(', '.join(insights['top_error_products']) or 'none')}.</li></ul><h3>Recommendations</h3>{_list(recommendations)}</section>
<section><h2>11. Conclusion and Limitations</h2><p>The pipeline provides a reproducible view of historical sales, chronological model validation, final-test evaluation, product-level model selection, and future estimates. The historical period is limited to 24 months, the final test period to three months, and the models primarily use time-index features rather than external drivers. Forecast generation is an analytical output and does not represent production deployment.</p></section>
<footer>Generated locally by <code>python main.py all</code>. Source artifacts remain available in the configured temporary output directory.</footer></main></body></html>"""
    report_path = report_dir / "index.html"
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path
