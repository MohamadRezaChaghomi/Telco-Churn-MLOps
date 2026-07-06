from pathlib import Path

import pandas as pd

from src.preprocess import preprocess_data
from src.feature_engineering import engineer_features


def test_preprocess_data_creates_valid_output():
    raw_path = Path("data/v1/Telco_customer_churn (1).xlsx")
    raw_df = pd.read_excel(raw_path)

    cleaned_df = preprocess_data(raw_df)

    assert "Churn Value" in cleaned_df.columns
    assert cleaned_df["Total Charges"].isna().sum() == 0
    assert cleaned_df.shape[0] == raw_df.shape[0]
    assert cleaned_df["Contract"].dtype in ["int64", "int32", "int16"]


def test_engineer_features_adds_expected_columns():
    raw_path = Path("data/v1/Telco_customer_churn (1).xlsx")
    raw_df = pd.read_excel(raw_path)
    cleaned_df = preprocess_data(raw_df)
    engineered_df = engineer_features(cleaned_df)

    assert "Avg Monthly Charge" in engineered_df.columns
    assert "Tenure Group" in engineered_df.columns
    assert "Has Internet & Phone" in engineered_df.columns
    assert "Interaction_Contract_Tenure" in engineered_df.columns
