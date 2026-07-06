# Final Report: Telco Customer Churn Prediction

## 1. Project Objective
The goal of this project was to build a complete MLOps workflow for predicting customer churn using the Telco customer churn Excel dataset.

## 2. Data Source
- File: data/v1/Telco_customer_churn (1).xlsx
- Rows: 7,043
- Key target column: Churn Value

## 3. Implementation Workflow
1. Data loading from Excel
2. Data cleaning and preprocessing
3. Feature engineering
4. Training and evaluation of multiple models
5. Experiment tracking with MLflow
6. Saving the best model for deployment
7. Prediction API using Flask
8. Containerization with Docker

## 4. Models Compared
- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

## 5. Best Model Result
- Best model: Random Forest
- F1 score: 0.8639
- ROC-AUC: 0.9738

## 6. Deliverables
- Cleaned dataset: data/v2/telco_churn_clean.csv
- Feature-engineered dataset: data/v3/telco_churn_featured.csv
- Trained model: models/best_model.joblib
- Feature schema: models/feature_columns.json
- Prediction API: docker/app.py
- Container definition: docker/Dockerfile

## 7. How to Run
```bash
python -m src.run_pipeline
pytest -q
```

## 8. Submission Note
This project is fully organized, executable, and prepared for final presentation to the instructor.
