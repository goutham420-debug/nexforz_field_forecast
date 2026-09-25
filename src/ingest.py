from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/polyhouse_sensors.csv")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

# Generate synthetic sample if no raw data exists yet
if not RAW.exists():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 365
    temp = rng.normal(22, 1.5, n)
    hum = np.clip(rng.normal(87, 3, n), 75, 98)
    co2 = rng.normal(900, 80, n)
    # Simple synthetic yield relationship
    yield_kg = 8 + 0.3 * temp + 0.05 * hum - 0.002 * co2 + rng.normal(0, 0.5, n)

    df_raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "temperature_c": temp.round(2),
            "humidity_pct": hum.round(1),
            "co2_ppm": co2.round(0),
            "yield_kg": yield_kg.round(2),
        }
    )
    df_raw.to_csv(RAW, index=False)
    print(f"Generated raw dataset: {RAW}")

# Ingest polyhouse sensor CSV
df = pd.read_csv(
    RAW,
    parse_dates=["timestamp"],
    dtype={
        "temperature_c": "float64",
        "humidity_pct": "float64",
        "co2_ppm": "float64",
        "yield_kg": "float64",
    },
)

print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nHead:\n", df.head())

df.to_parquet(INTERIM / "01_loaded.parquet", index=False)
print("\nSaved snapshot to data/interim/01_loaded.parquet")