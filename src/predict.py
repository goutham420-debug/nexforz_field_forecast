import json
from pathlib import Path
import joblib
import pandas as pd

MODEL_DIR = Path("models")

# Load artifacts
_scaler = joblib.load(MODEL_DIR / "scaler.joblib")
_model = joblib.load(MODEL_DIR / "champion.joblib")
with open(MODEL_DIR / "feature_cols.json", "r") as f:
    _feature_cols = json.load(f)

# Raw feature names expected before scaling
RAW_COLS = ["temperature_c", "humidity_pct", "co2_ppm", "temp_humid_interaction"]

def predict_yield(temperature_c: float, humidity_pct: float, co2_ppm: float) -> float:
    """Predicts polyhouse yield (kg) given raw environmental readings."""
    temp_humid_interaction = temperature_c * humidity_pct / 100.0

    raw_df = pd.DataFrame(
        [[temperature_c, humidity_pct, co2_ppm, temp_humid_interaction]],
        columns=RAW_COLS
    )

    # Scale using persisted scaler fit on train
    scaled_values = _scaler.transform(raw_df)
    scaled_df = pd.DataFrame(scaled_values, columns=_feature_cols)

    # Predict using champion model
    prediction = _model.predict(scaled_df)
    return float(prediction[0])

if __name__ == "__main__":
    sample_temp = 22.0
    sample_humidity = 88.0
    sample_co2 = 920.0

    predicted_kg = predict_yield(sample_temp, sample_humidity, sample_co2)
    print("--- Inference Helper Verification ---")
    print(f"Input: Temp={sample_temp}°C, Humidity={sample_humidity}%, CO2={sample_co2} ppm")
    print(f"Predicted Polyhouse Yield: {predicted_kg:.2f} kg")