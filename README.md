# Marketplace Sales Performance Analysis

This project analyzes historical marketplace sales and evaluates time-series forecasting strategies using the CRISP-DM methodology.

## Project overview

- Historical period: January 2024 to December 2025
- Forecast period: January to March 2026
- Products: 20
- Cleaned transactions: 41,954
- Monthly modeling observations: 480 product-month rows
- Forecasting strategies: 12
- Walk-forward validation folds: 3

The pipeline preserves chronological order throughout the experiment. Random splitting is not used because it could expose future observations during training.

## Pipeline execution

Install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Run the complete workflow with one command:

```bash
python main.py all
```

The pipeline performs:

1. Data cleaning and relational integration
2. Monthly aggregation and chronological splitting
3. Model training and walk-forward validation
4. Final-test evaluation
5. Future forecasting
6. Generation of eight visualizations
7. Deterministic business insights and error analysis
8. Local HTML report generation

The final report is written to:

```text
reports/latest/index.html
```

Individual commands are also available:

```bash
python main.py clean
python main.py model
python main.py visualize
python main.py insights
python main.py report
```

All commands support `--data-dir`, `--out-dir`, and `--config`. Chart and report locations can be changed with `--visualizations-dir` and `--report-dir`.

## Architecture

```text
sales-performance-analysis/
├── datasets/
│   ├── orders.csv
│   ├── customers.csv
│   ├── payments.csv
│   └── products.csv
├── salespipe/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── config.py
│   ├── modeling.py
│   ├── visualization.py
│   ├── insights.py
│   └── reporting.py
├── temp/
├── visualizations/
├── reports/latest/
├── main.py
├── config.example.json
├── requirements.txt
└── README.md
```

### Module responsibilities

- `cleaning.py`: validates, cleans, merges, and feature-engineers raw datasets.
- `config.py`: loads schema, model, and period configuration.
- `modeling.py`: builds monthly data, trains models, performs validation, evaluates strategies, and forecasts future sales.
- `visualization.py`: generates the eight analytical charts.
- `insights.py`: creates reproducible, rule-based findings from generated artifacts.
- `reporting.py`: creates the local HTML report and its chart assets.
- `main.py`: orchestrates the complete workflow and command-line interface.

## Data preparation

The cleaning workflow:

1. Reads the four raw tables.
2. Validates required columns.
3. Removes duplicate orders.
4. Converts dates and numeric fields.
5. Removes invalid dates, quantities, and discounts.
6. Imputes missing customer ages with the median.
7. Normalizes city names and missing categorical values.
8. Validates and performs relational merges.
9. Calculates transaction sales:

```text
Sales = Quantity × UnitPrice × (1 − Discount)
```

Only completed transactions are used for monthly sales modeling.

## Modeling methodology

The target is monthly product sales. The model uses a chronological month index and evaluates 12 regression and forecasting strategies:

1. Training Mean Baseline
2. Linear Regression
3. Polynomial Regression, degree 2
4. Polynomial Regression, degree 5
5. Ridge Regression, alpha 0.1
6. Ridge Regression, alpha 1.0
7. Ridge Regression, alpha 10.0
8. Lasso Regression, alpha 0.1
9. Lasso Regression, alpha 1.0
10. Lasso Regression, alpha 10.0
11. Seasonal Naive
12. Moving Average

### Chronological split

| Dataset | Period | Months |
|---|---|---:|
| Training | January 2024 – May 2025 | 17 |
| Validation | June 2025 – September 2025 | 4 |
| Testing | October 2025 – December 2025 | 3 |

### Walk-forward validation

| Fold | Training period | Validation period |
|---:|---|---|
| 1 | January–December 2024 | January–March 2025 |
| 2 | January 2024–March 2025 | April–June 2025 |
| 3 | January 2024–June 2025 | July–September 2025 |

Models are selected using validation RMSE. The final test period remains untouched until the final comparison.

## Evaluation

The final evaluation compares:

1. **Global Training Mean Baseline**
2. **Best Global ML**
3. **Per-Product Selection**

Current reference results:

| Strategy | MAE | RMSE | WAPE | R² |
|---|---:|---:|---:|---:|
| Global Training Mean Baseline | 769.89 | 1,101.98 | 13.59% | 0.858 |
| Best Global ML | 818.86 | 1,173.88 | 14.46% | 0.839 |
| Per-Product Selection | 928.47 | 1,326.27 | 16.39% | 0.794 |

These values are reference results only. The generated artifacts and report calculate metrics dynamically.

## Generated visualizations

The pipeline creates:

1. Overall monthly sales
2. Monthly sales trends by product
3. Total sales by product
4. Walk-forward model performance comparison
5. Actual versus predicted sales
6. Forecasting error contribution
7. Overall future sales forecast
8. Future sales forecast by product

## Reproducibility and limitations

The pipeline preserves raw data and writes intermediate CSV artifacts for verification. Business insights are deterministic Python calculations rather than LLM-generated interpretations.

Limitations include the 24-month historical period, the three-month final test period, the 20-product scope, and the limited use of external explanatory variables. Forecasts are analytical estimates and do not represent production deployment.
