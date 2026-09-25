from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/interim/01_loaded.parquet")

print("--- Missing Value Report Before Fixes ---")
print(df.isna().sum())

# Agritech operational thresholds
valid = (
    df["humidity_pct"].between(50, 100)
    & df["temperature_c"].between(10, 35)
    & df["co2_ppm"].between(400, 2000)
    & df["yield_kg"].notna()
)
df = df[valid].copy()

# Forward fill sensor short dropouts
cols = ["temperature_c", "humidity_pct", "co2_ppm"]
df[cols] = df[cols].ffill(limit=2)

# Drop missing targets and timestamp duplicates
df = df.dropna(subset=["yield_kg"])
df = df.drop_duplicates(subset=["timestamp"], keep="last")

output_path = PROCESSED / "02_cleaned.parquet"
df.to_parquet(output_path, index=False)

print("\n--- Cleaning Summary ---")
print(f"Clean rows: {len(df)}")
print(f"Cleaned dataset saved to: {output_path}")