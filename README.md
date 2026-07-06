# Telco Customer Churn - MLOps Project

This repository implements an end-to-end MLOps workflow for predicting customer churn using the Telco Customer Churn dataset stored in the Excel file under [data/v1](data/v1). The project covers data ingestion, preprocessing, feature engineering, model training, evaluation, experiment tracking with MLflow, and deployment as a Flask API inside Docker.

## Project goals
- Load the raw Excel data from [data/v1](data/v1).
- Create cleaned and feature-engineered datasets in [data/v2](data/v2) and [data/v3](data/v3).
- Train and compare multiple classification models.
- Track experiments and artifacts with MLflow.
- Expose a prediction API and containerize it with Docker.

## Tech stack
- Python 3.10+
- pandas, NumPy, scikit-learn
- XGBoost and CatBoost
- MLflow
- Flask
- Docker
- pytest

## Data source
- Raw file: [data/v1](data/v1)
- Dataset size: 7,043 rows
- Target column: Churn Value

## Repository structure
- [src](src): pipeline modules for data loading, preprocessing, feature engineering, training, evaluation, and orchestration.
- [data/v2](data/v2): cleaned dataset ready for modeling.
- [data/v3](data/v3): feature-engineered dataset.
- [models](models): trained model, feature schema, and evaluation artifacts.
- [reports](reports): comparison reports and final project documentation.
- [docker](docker): Flask API and Docker deployment files.

## Setup
On Windows, run the following commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the training pipeline
```powershell
python -m src.run_pipeline
```

This will:
1. Load the raw Excel data.
2. Save a cleaned dataset to [data/v2](data/v2).
3. Create engineered features and save them to [data/v3](data/v3).
4. Train multiple models and save the best model to [models](models).

## Run tests
```powershell
pytest -q
```

## Run the prediction API locally
```powershell
python docker/app.py
```

The API exposes two endpoints:
- GET /health
- POST /predict

Example request body:
```json
{
  "features": [0, 1, 0, 12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 45.5, 546.0, 0.72, 350, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

The feature list must follow the same order as [models/feature_columns.json](models/feature_columns.json).

## MLflow tracking
To view experiment runs locally:

```powershell
mlflow ui --port 5001
```

Then open http://127.0.0.1:5001 in your browser.

## Docker deployment
Build and run the container:

```powershell
docker build -t churn-predictor -f docker/Dockerfile .
docker run -d --name churn-api -p 5000:5000 churn-predictor
```

Verify the service:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/health -Method Get
```

## Main results
- Best model: Random Forest
- F1 score: 0.8635
- ROC-AUC: 0.9686
- Validation with pytest: 2 passed

## Model comparison
See [reports/model_comparison.md](reports/model_comparison.md) for the full comparison table.

Random Forest was selected as the final model because it achieved the highest F1-score among the evaluated classifiers, which is the most relevant metric for this imbalanced churn prediction task.

## Deliverables
- Cleaned dataset: [data/v2](data/v2)
- Feature-engineered dataset: [data/v3](data/v3)
- Trained model: [models/best_model.joblib](models/best_model.joblib)
- Feature schema: [models/feature_columns.json](models/feature_columns.json)
- Final report: [reports/final_report.md](reports/final_report.md)
