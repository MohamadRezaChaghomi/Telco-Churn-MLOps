import json
import urllib.request
from pathlib import Path

import pandas as pd

root = Path(r'C:\Users\AsuS\Desktop\project\Telco-Churn-MLOps\Telco-Churn-MLOps')
features = json.loads((root / 'models' / 'feature_columns.json').read_text(encoding='utf-8'))
df = pd.read_csv(root / 'data' / 'v3' / 'telco_churn_featured.csv')
payload = {'features': df[features].iloc[0].tolist()}
req = urllib.request.Request(
    'http://127.0.0.1:5000/predict',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'},
)
with urllib.request.urlopen(req, timeout=20) as response:
    print(response.read().decode())
