# Telco Customer Churn - MLOps Project

This project implements a complete and production-style MLOps workflow for predicting customer churn using the Telco Customer Churn dataset from the Excel file stored in the v1 folder.

## Project goals
- Load the raw Excel data from the v1 folder.
- Create cleaned and feature-engineered versions in v2 and v3.
- Train multiple classification models and compare them.
- Log experiments with MLflow.
- Deploy a prediction service with Flask and Docker.

## Data source
- Raw data file: data/v1/Telco_customer_churn (1).xlsx
- Dataset size: 7,043 rows
- Target column: Churn Value

## Main results
- Best model: Random Forest
- F1 score: 0.8635
- ROC-AUC: 0.9686
- Verified with: pytest (2 passed)

## Model comparison
See [reports/model_comparison.md](reports/model_comparison.md) for the full comparison table.

Random Forest was chosen as the final model because it achieved the highest F1-score among the evaluated classifiers, which is the most relevant metric for this imbalanced churn prediction task.

## Run the pipeline
```bash
python -m src.run_pipeline
```

## Run tests
```bash
pytest -q
```

## Run the prediction API
```bash
python docker/app.py
```

## Docker
```bash
docker build -t telco-churn-api -f docker/Dockerfile .
docker run -p 5000:5000 telco-churn-api
```

## Deliverables
- Cleaned dataset: data/v2/telco_churn_clean.csv
- Feature-engineered dataset: data/v3/telco_churn_featured.csv
- Trained model: models/best_model.joblib
- Feature schema: models/feature_columns.json
- Final report: reports/final_report.md
