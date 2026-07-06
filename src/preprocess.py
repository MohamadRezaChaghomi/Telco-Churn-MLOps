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


def _encode_yes_no(series: pd.Series) -> pd.Series:
    return series.replace({"No": 0, "Yes": 1, "No internet service": 0, "No phone service": 0}).fillna(0).astype(int)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco dataset and prepare it for modeling."""
    data = df.copy()
    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns], errors="ignore")

    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce").fillna(0)
    data["Monthly Charges"] = pd.to_numeric(data["Monthly Charges"], errors="coerce")

    for col in [
        "Phone Service",
        "Multiple Lines",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Paperless Billing",
    ]:
        if col in data.columns:
            data[col] = _encode_yes_no(data[col])

    if "Senior Citizen" in data.columns:
        data["Senior Citizen"] = data["Senior Citizen"].replace({"No": 0, "Yes": 1}).fillna(0).astype(int)

    if "Partner" in data.columns:
        data["Partner"] = data["Partner"].replace({"No": 0, "Yes": 1}).fillna(0).astype(int)

    if "Dependents" in data.columns:
        data["Dependents"] = data["Dependents"].replace({"No": 0, "Yes": 1}).fillna(0).astype(int)

    if "Gender" in data.columns:
        data["Gender"] = data["Gender"].replace({"Female": 1, "Male": 0}).fillna(0).astype(int)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    if "Contract" in data.columns:
        data["Contract"] = data["Contract"].map(contract_map).fillna(0).astype(int)

    if "Internet Service" in data.columns:
        data["Internet Service"] = data["Internet Service"].replace({"No": 0, "DSL": 1, "Fiber optic": 2}).fillna(0).astype(int)

    payment_method_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3,
    }
    if "Payment Method" in data.columns:
        data["Payment Method"] = data["Payment Method"].map(payment_method_map).fillna(0).astype(int)

    if TARGET_COLUMN in data.columns:
        data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    return data
