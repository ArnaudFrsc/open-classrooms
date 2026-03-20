import struct
from pathlib import Path
import os

MODEL_PATH = Path(r"C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P7\mlruns\LightGBM_best_model.pkl")

import joblib
model = joblib.load(MODEL_PATH)
print(type(model))