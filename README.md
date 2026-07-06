# Telco Customer Churn - MLOps Project

This project implements a complete MLOps workflow for predicting customer churn using the Telco Customer Churn dataset from the Excel file stored in the v1 folder.

## Project goals
- Load the raw Excel data from the v1 folder.
- Create cleaned and feature-engineered versions in v2 and v3.
- Train multiple classification models and compare them.
- Log experiments with MLflow.
- Deploy a prediction service with Flask and Docker.

## Data source
- Raw data file: data/v1/Telco_customer_churn (1).xlsx
- Dataset size: 7043 rows
- Target column: Churn Value

## Main results
- Best model: Random Forest
- F1 score: 0.8639
- ROC-AUC: 0.9738

## Run the pipeline
```bash
python -m src.run_pipeline
```

## Run tests
```bash
pytest -q
```

## Deployment
The trained model is saved in models/best_model.joblib and can be served with the Flask app in docker/app.py.
