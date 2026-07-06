from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.evaluate import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _build_model(name: str):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=42)
    if name == "random_forest":
        return RandomForestClassifier(random_state=42)
    if name == "xgboost":
        return XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=100)
    if name == "catboost":
        return CatBoostClassifier(verbose=False, random_state=42, iterations=200)
    raise ValueError(f"Unknown model: {name}")


def _build_pipeline(model_name: str) -> Pipeline:
    model = _build_model(model_name)
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def train_models(X: pd.DataFrame, y: pd.Series, experiment_name: str = "telco-churn") -> dict:
    """Train multiple models and save the best candidate to disk."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    tracking_uri = (PROJECT_ROOT / "mlruns").resolve().as_uri()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results = {}
    best_score = -1.0
    best_model_name = ""
    best_pipeline = None
    best_metrics = None

    for model_name in ["logistic_regression", "random_forest", "xgboost", "catboost"]:
        with mlflow.start_run(run_name=model_name):
            if model_name == "random_forest":
                param_grid = {"model__n_estimators": [50, 100], "model__max_depth": [5, 10]}
                pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(random_state=42))])
                search = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1)
                search.fit(X_train, y_train)
                pipeline = search.best_estimator_
            else:
                pipeline = _build_pipeline(model_name)
                pipeline.fit(X_train, y_train)

            val_metrics = evaluate_model(pipeline, X_val, y_val)
            test_metrics = evaluate_model(pipeline, X_test, y_test)

            mlflow.log_params({"model": model_name, "random_state": 42})
            mlflow.log_metrics({**{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}, **{f"test_{k}": v for k, v in test_metrics.items() if k != "confusion_matrix"}})
            trusted_types = []
            if model_name == "xgboost":
                trusted_types = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
            if model_name == "catboost":
                trusted_types = ["catboost.core.CatBoostClassifier"]
            mlflow.sklearn.log_model(pipeline, artifact_path=model_name, skops_trusted_types=trusted_types)

            results[model_name] = {"val": val_metrics, "test": test_metrics}
            if test_metrics["f1"] > best_score:
                best_score = test_metrics["f1"]
                best_model_name = model_name
                best_pipeline = pipeline
                best_metrics = test_metrics

    if best_pipeline is None:
        raise RuntimeError("Training failed to produce a model")

    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")
    feature_names = list(X.columns)
    with (MODELS_DIR / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    mlflow.set_tag("best_model", best_model_name)
    mlflow.log_metric("best_test_f1", best_score)

    return {
        "best_model": best_model_name,
        "best_metrics": best_metrics,
        "all_results": results,
    }
