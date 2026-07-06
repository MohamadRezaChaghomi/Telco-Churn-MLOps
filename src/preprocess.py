from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "Churn Value"
DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Reason",
]

YES_NO_COLUMNS = [
    "Phone Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Paperless Billing",
]


def _encode_yes_no(series: pd.Series) -> pd.Series:
    return series.map({"No": 0, "Yes": 1, "No internet service": 0, "No phone service": 0}).fillna(0).astype(int)


def _clip_outliers(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco dataset and prepare it for modeling."""
    data = df.copy()
    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns], errors="ignore")

    for col in ["Total Charges", "Monthly Charges", "Tenure Months"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "Total Charges" in data.columns:
        data["Total Charges"] = data["Total Charges"].fillna(0)
    if "Monthly Charges" in data.columns:
        data["Monthly Charges"] = data["Monthly Charges"].fillna(data["Monthly Charges"].median())
    if "Tenure Months" in data.columns:
        data["Tenure Months"] = data["Tenure Months"].fillna(data["Tenure Months"].median())

    for col in ["Monthly Charges", "Total Charges", "Tenure Months"]:
        if col in data.columns:
            data[col] = _clip_outliers(data[col])

    for col in YES_NO_COLUMNS:
        if col in data.columns:
            data[col] = _encode_yes_no(data[col])

    if "Senior Citizen" in data.columns:
        data["Senior Citizen"] = data["Senior Citizen"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    if "Partner" in data.columns:
        data["Partner"] = data["Partner"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    if "Dependents" in data.columns:
        data["Dependents"] = data["Dependents"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    if "Contract" in data.columns:
        data["Contract"] = data["Contract"].map(contract_map).fillna(0).astype(int)

    for col in data.columns:
        if col in {TARGET_COLUMN, "Contract"}:
            continue
        if data[col].dtype == "object" or pd.api.types.is_categorical_dtype(data[col]):
            data = pd.concat([data.drop(columns=[col]), pd.get_dummies(data[col], prefix=col, dtype=int)], axis=1)

    if TARGET_COLUMN in data.columns:
        data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    return data
