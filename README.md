# Telco Customer Churn - MLOps Project

This project implements a complete MLOps workflow for predicting customer churn using the Telco Customer Churn dataset.

## Project goals
- Load the raw Excel data from the v1 folder.
- Create cleaned and feature-engineered versions in v2 and v3.
- Train multiple classification models and compare them.
- Log experiments with MLflow.
- Deploy a prediction service with Flask and Docker.

## Run the pipeline
```bash
python -m src.run_pipeline
```

## Run tests
```bash
pytest -q
```
