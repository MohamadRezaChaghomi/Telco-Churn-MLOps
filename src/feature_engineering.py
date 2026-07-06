from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional engineered features for the churn dataset."""
    data = df.copy()
    data["Avg Monthly Charge"] = data["Total Charges"] / (data["Tenure Months"] + 1)

    bins = [-np.inf, 12, 36, np.inf]
    labels = ["Short", "Medium", "Long"]
    data["Tenure Group"] = pd.cut(data["Tenure Months"], bins=bins, labels=labels, include_lowest=True)
    data["Tenure Group"] = data["Tenure Group"].astype("object")

    has_internet = data.get("Internet Service", 0).astype(int) > 0
    has_phone = data.get("Phone Service", 0).astype(int) > 0
    data["Has Internet & Phone"] = (has_internet & has_phone).astype(int)

    data["Interaction_Contract_Tenure"] = data["Contract"] * data["Tenure Months"]

    tenure_group_map = {"Short": 0, "Medium": 1, "Long": 2}
    data["Tenure Group"] = data["Tenure Group"].map(tenure_group_map).fillna(0).astype(int)
    return data
