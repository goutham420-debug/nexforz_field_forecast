from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Ensure target directories exist
Path("models").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Load data
processed_input = Path("data/processed/02_cleaned.parquet")
interim_input = Path("data/interim/02_cleaned.parquet")

if processed_input.exists():
    df = pd.read_parquet(processed_input)
elif interim_input.exists():
    df = pd.read_parquet(interim_input)
else:
    raise FileNotFoundError("Could not find 02_cleaned.parquet!")

# 1. Chronological Sorting
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 2. Domain-informed Interaction Feature
df["temp_humid_interaction"] = df["temperature_c"] * df["humidity_pct"] / 100

feature_cols = ["temperature_c", "humidity_pct", "co2_ppm", "temp_humid_interaction"]
target_col = "yield_kg"

# 3. Temporal Train/Test Split (80/20 chronological split - No Leakage)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

# 4. Fit MinMaxScaler on TRAIN SET ONLY
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Transform test without fitting!

# 5. Save scaler to models/scaler.joblib
scaler_path = "models/scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"Saved fitted scaler to: {scaler_path}")

# 6. Save split datasets to data/processed/
scaled_cols = [c + "_scaled" for c in feature_cols]

train_scaled_df = pd.DataFrame(X_train_scaled, columns=scaled_cols, index=train_df.index)
train_scaled_df["timestamp"] = train_df["timestamp"]
train_scaled_df[target_col] = y_train.values

test_scaled_df = pd.DataFrame(X_test_scaled, columns=scaled_cols, index=test_df.index)
test_scaled_df["timestamp"] = test_df["timestamp"]
test_scaled_df[target_col] = y_test.values

train_scaled_df.to_parquet("data/processed/train.parquet", index=False)
test_scaled_df.to_parquet("data/processed/test.parquet", index=False)

# Full feature set for downstream tasks
full_scaled = pd.concat([train_scaled_df, test_scaled_df], axis=0)
full_scaled.to_parquet("data/processed/features.parquet", index=False)

# Print split summary
print("\n=== TEMPORAL SPLIT SUMMARY ===")
print(f"Total rows: {len(df)}")
print(f"Train rows: {len(train_df)} | Date range: {train_df['timestamp'].min().strftime('%Y-%m-%d')} to {train_df['timestamp'].max().strftime('%Y-%m-%d')}")
print(f"Test rows:  {len(test_df)} | Date range: {test_df['timestamp'].min().strftime('%Y-%m-%d')} to {test_df['timestamp'].max().strftime('%Y-%m-%d')}")
print(f"Saved: models/scaler.joblib, data/processed/train.parquet, data/processed/test.parquet")