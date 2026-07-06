from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_raw_data, save_dataset
from src.feature_engineering import engineer_features
from src.preprocess import preprocess_data
from src.train import train_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the churn prediction MLOps pipeline")
    parser.add_argument("--version", default="v3", choices=["v1", "v2", "v3"])
    args = parser.parse_args()

    raw_df = load_raw_data()
    cleaned_df = preprocess_data(raw_df)
    save_dataset(cleaned_df, version="v2", filename="telco_churn_clean.csv")

    engineered_df = engineer_features(cleaned_df)
    save_dataset(engineered_df, version="v3", filename="telco_churn_featured.csv")

    target_col = "Churn Value"
    X = engineered_df.drop(columns=[target_col], errors="ignore")
    y = engineered_df[target_col]

    result = train_models(X, y, experiment_name=f"telco-churn-{args.version}")
    print(f"Best model: {result['best_model']}")
    print(result["best_metrics"])


if __name__ == "__main__":
    main()
