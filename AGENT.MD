
# Task: Refactor and Build an End-to-End Machine Learning Sales Analysis Pipeline

You are a senior Python software engineer and machine learning engineer. Your task is to refactor my existing Python machine learning project into a clean, modular, maintainable, end-to-end analytical pipeline.

## 1. Project Context

This project is an academic machine learning research project titled:

**Analisis Kinerja Penjualan Marketplace Online Berbasis Machine Learning Menggunakan Kerangka CRISP-DM**

The project analyzes historical e-commerce sales data and applies machine learning and forecasting techniques to evaluate sales performance and forecast future sales.

The project follows the CRISP-DM methodology.

The existing implementation already contains:
- Data cleaning and dataset integration.
- Exploratory sales analysis.
- Multiple forecasting models.
- Chronological train, validation, and test splits.
- Walk-forward validation.
- Model evaluation.
- Product-level model selection.
- Future sales forecasting.
- Forecasting error analysis.
- Eight existing Matplotlib visualizations.

Your first responsibility is to inspect the existing codebase and understand its implementation before making changes.

Do not assume that the project structure or implementation is exactly as described here. Inspect the actual files.

## 2. Primary Objective

Build a unified pipeline that can execute the entire analytical workflow through a single command:

`python main.py all`

The pipeline must:

1. Clean and prepare datasets.
2. Train and evaluate forecasting models.
3. Perform walk-forward validation.
4. Generate future sales forecasts.
5. Generate visualizations.
6. Generate analytical findings and business insights.
7. Generate a comprehensive HTML report.

The final report should be generated at:

`reports/latest/index.html`

## 3. Target Project Architecture

Refactor the existing code into the following modular structure:

```text
sales-performance-analysis/
│
├── datasets/
│
├── salespipe/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── config.py
│   ├── modeling.py
│   ├── visualization.py
│   ├── insights.py
│   └── reporting.py
│
├── temp/
├── visualizations/
├── reports/
│   └── latest/
│       ├── index.html
│       └── assets/
│
├── main.py
├── config.example.json
├── requirements.txt
├── README.md
└── .gitignore
```

Preserve existing modules where appropriate. Do not unnecessarily rewrite working implementations.

### Module responsibilities

**cleaning.py**
- Clean raw datasets.
- Handle missing values and duplicates.
- Validate data.
- Merge datasets.
- Perform existing feature engineering.

**config.py**
- Load project configuration.
- Keep configurable parameters centralized.
- Avoid hard-coded paths and experimental settings where possible.

**modeling.py**
- Train forecasting models.
- Perform walk-forward validation.
- Evaluate models.
- Select models.
- Generate future forecasts.
- Preserve existing modeling methodology.

**visualization.py**
- Generate all eight existing visualizations.
- Accept input and output directories as function arguments.
- Avoid hard-coded global file paths.
- Return generated image paths.
- Close Matplotlib figures after saving.
- Keep chart calculations and analytical logic unchanged.

**insights.py**
- Analyze model performance.
- Identify important sales trends.
- Identify products contributing most to forecasting errors.
- Compare forecasting strategies.
- Generate data-driven business insights and recommendations.
- Return structured results that can be consumed by the reporting module.

**reporting.py**
- Generate the final HTML report.
- Include analytical tables, charts, insights, and recommendations.
- Generate report assets when necessary.
- Use relative asset paths so the report works locally.
- Make the report readable and professional.

**main.py**
- Serve as the pipeline entry point.
- Coordinate the modules.
- Handle CLI arguments.
- Configure logging.
- Handle errors.
- Display pipeline progress.

## 4. Visualization Requirements

I already have eight working visualizations in `visualize_sales.py`.

You must inspect and migrate the existing implementation into `salespipe/visualization.py`.

Preserve all eight visualizations:

1. Overall Monthly Sales.
2. Monthly Sales Trends by Product.
3. Total Sales by Product.
4. Walk-Forward Model Performance Comparison.
5. Actual vs Predicted Sales.
6. Top 10 Products Contributing to Forecasting Error.
7. Overall Future Sales Forecast.
8. Future Sales Forecast by Product.

Do not redesign or remove these charts without a clear technical reason.

Refactor the implementation to:
- Use configurable input and output directories.
- Centralize CSV loading.
- Centralize figure saving.
- Return generated chart paths.
- Avoid filesystem side effects during module imports.
- Maintain type annotations.
- Remain compatible with Pyright.

## 5. Business Insights Requirements

Implement deterministic, rule-based business insights using Python.

The insights should be generated from actual project artifacts and metrics.

Examples:
- Identify the best-performing forecasting strategy based on final test metrics.
- Compare model performance across walk-forward validation folds.
- Identify products with increasing or decreasing sales trends.
- Identify products contributing most to forecasting errors.
- Identify products with the largest absolute and relative prediction errors.
- Summarize future forecast trends.
- Generate recommendations based on the observed results.

Important:
- Do not invent business findings.
- Do not make unsupported causal claims.
- Distinguish descriptive findings from recommendations.
- Do not use an LLM to fabricate or interpret metrics.
- Handle edge cases such as missing data and zero denominators.

## 6. HTML Report Requirements

Generate a professional analytical report at:

`reports/latest/index.html`

The report should contain:

1. Executive Summary
2. Business Understanding
3. Data Understanding
4. Data Preparation
5. Modeling Methodology
6. Model Evaluation
7. Sales Performance Analysis
8. Forecasting Results
9. Business Insights and Recommendations
10. Conclusion
11. Project Limitations

Include:
- Key performance indicators.
- Dataset statistics.
- Model comparison tables.
- Walk-forward validation results.
- Final test evaluation metrics.
- Sales analysis charts.
- Forecast charts.
- Product-level analysis.
- Error analysis.
- Business insights.

Use a clean, modern, professional design suitable for an academic machine learning research project.

The report must work locally without requiring a web server or external API.

Do not claim that the model has been deployed into production. Forecast generation and report generation are analytical outputs, not production deployment.

## 7. CLI Requirements

Preserve the existing commands:

```bash
python main.py clean
python main.py model
python main.py all
```

Add commands if useful:

```bash
python main.py visualize
python main.py report
```

Preserve existing arguments:

```bash
--data-dir
--out-dir
--config
```

The `all` command must execute the entire pipeline in the correct dependency order.

Avoid duplicating execution logic across commands.

## 8. Engineering Standards

Use modern Python best practices.

Requirements:
- Python 3.10+.
- Type annotations.
- Clear function and module responsibilities.
- PEP 8 compliance.
- Descriptive naming.
- Appropriate logging.
- Explicit error handling.
- pathlib for filesystem operations.
- Avoid unnecessary global state.
- Avoid unnecessary dependencies.
- Keep the code compatible with Ruff and Pyright.
- Use pandas and Matplotlib consistently with the existing implementation.

Do not introduce unnecessary design patterns, abstractions, or frameworks.

Prefer straightforward, readable code over excessive abstraction.

## 9. Critical Data and Modeling Constraints

The existing experiment has already produced results.

You must preserve the existing data preparation and modeling methodology unless an actual implementation bug is discovered.

Important project facts:

- Historical dataset period: January 2024 to December 2025.
- Final cleaned dataset: 41,954 rows.
- Monthly dataset: 480 rows, representing 20 products across 24 months.
- Models and baselines evaluated: 12.
- Walk-forward validation: 3 chronological folds.
- Original training period: January 2024 to May 2025.
- Validation period: June to September 2025.
- Test period: October to December 2025.
- Future forecast period: January to March 2026.

Final test comparison:

| Strategy | MAE | RMSE | WAPE | R² |
|---|---:|---:|---:|---:|
| Global Naive | 769.89 | 1101.98 | 13.59% | 0.858 |
| Best Global ML (Ridge alpha 10) | 818.86 | 1173.88 | 14.46% | 0.839 |
| Per-Product Selection | 928.47 | 1326.27 | 16.39% | 0.794 |

Global Naive achieved the best aggregate final test performance.

Do not hard-code these metrics into the report. Read the actual generated CSV artifacts and calculate the findings dynamically. These figures are reference values for checking whether the existing implementation remains consistent.

## 10. Data Integrity and Reproducibility

- Preserve raw datasets.
- Do not overwrite source CSV files.
- Do not delete existing analytical artifacts.
- Do not change model parameters without justification.
- Maintain chronological validation to prevent data leakage.
- Ensure that all report findings are traceable to generated artifacts.
- Keep intermediate outputs available for academic verification.

## 11. Implementation Workflow

Work incrementally.

**Phase 1: Inspect**
- Inspect the entire project structure.
- Read all existing Python modules.
- Understand the current data flow and artifact dependencies.
- Identify existing functionality that should be preserved.
- Present a concise implementation plan.

**Phase 2: Refactor Visualization**
- Migrate all eight visualizations.
- Introduce configurable paths.
- Implement shared helpers.
- Verify all charts are generated correctly.

**Phase 3: Implement Insights**
- Create deterministic business analysis functions.
- Generate structured insights from existing artifacts.
- Verify numerical calculations.

**Phase 4: Implement Reporting**
- Generate the HTML report.
- Integrate tables, charts, and insights.
- Verify all report assets and links.

**Phase 5: Integrate Pipeline**
- Update `main.py`.
- Implement the complete `all` command.
- Preserve existing CLI functionality.

**Phase 6: Test**
- Run Ruff.
- Run Pyright.
- Run the existing pipeline.
- Run the complete pipeline.
- Verify generated CSV files, PNG charts, and HTML report.
- Check for missing files, incorrect paths, and inconsistent metrics.

After each phase:
1. Explain what was changed.
2. List modified and created files.
3. Run relevant checks.
4. Report any issues.
5. Wait for approval before proceeding to the next phase.

Do not make large, uncontrolled changes in a single step.

## 12. Final Acceptance Criteria

The project is complete when:

- `python main.py all` executes the complete workflow successfully.
- Existing cleaning and modeling results remain consistent.
- All eight visualizations are generated.
- Business insights are generated from actual analytical results.
- The HTML report contains all required sections.
- The report works locally.
- Existing CLI commands remain functional.
- Ruff and Pyright report no unresolved issues.
- No raw data or important intermediate artifacts are lost.

Start with Phase 1: inspect the repository and present your implementation plan. Do not modify files until the inspection and plan are presented.

