import requests
import pandas as pd

df = pd.DataFrame({
    "AMT_CREDIT": [500000, 250000],
    "AMT_INCOME_TOTAL": [150000, 80000],
    "DAYS_BIRTH": [-12000, -15000],
    "DAYS_EMPLOYED": [-3000, -500],
})

resp = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"data": df.to_dict(orient="records")},
)

predictions = resp.json()["predictions"]
df["default_probability"] = [p["default_probability"] for p in predictions]
print(df)