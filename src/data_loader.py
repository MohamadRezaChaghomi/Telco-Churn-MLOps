from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw Excel dataset from the v1 folder."""
    if path is None:
        path = PROJECT_ROOT / "data" / "v1" / "Telco_customer_churn (1).xlsx"
    return pd.read_excel(path)


def save_dataset(df: pd.DataFrame, version: str, filename: str, data_dir: str | None = None) -> Path:
    """Save a processed dataset to the requested version folder."""
    root = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / version
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / filename
    df.to_csv(output_path, index=False)
    return output_path


def load_processed_data(version: str, filename: str, data_dir: str | None = None) -> pd.DataFrame:
    """Load a processed dataset from disk."""
    root = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / version
    return pd.read_csv(root / filename)
