from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Ensure target directories exist
Path("models").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Check processed first, fallback to interim if needed
processed_input = Path("data/processed/02_cleaned.parquet")
interim_input = Path("data/interim/02_cleaned.parquet")

if processed_input.exists():
    df = pd.read_parquet(processed_input)
elif interim_input.exists():
    df = pd.read_parquet(interim_input)
else:
    raise FileNotFoundError("Could not find 02_cleaned.parquet in data/processed/ or data/interim/!")

# Sort chronologically by date
df = df.sort_values("timestamp")

# Domain-informed engineered interaction feature
df["temp_humid_interaction"] = df["temperature_c"] * df["humidity_pct"] / 100

feature_cols = ["temperature_c", "humidity_pct", "co2_ppm", "temp_humid_interaction"]
X = df[feature_cols]
y = df["yield_kg"]

# Min-Max Scaling on feature matrix
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Persist scaler object
scaler_path = "models/minmax_scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"Saved fitted scaler to: {scaler_path}")

# Construct scaled DataFrame and save to parquet
scaled_cols = [c + "_scaled" for c in feature_cols]
processed = pd.DataFrame(X_scaled, columns=scaled_cols)
processed["yield_kg"] = y.values

output_path = "data/processed/features.parquet"
processed.to_parquet(output_path, index=False)
print(f"Saved scaled features to: {output_path}")

# Verification checks
print("\n--- Feature Engineering & Scaling Verification ---")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Processed shape: {processed.shape}")
print(f"Any NaNs in processed: {processed.isna().any().any()}")
print("\nFeature min values (expected ~0.0):\n", processed[scaled_cols].min().round(3))
print("\nFeature max values (expected ~1.0):\n", processed[scaled_cols].max().round(3))
print("\nHead of processed features:\n", processed.head(3))