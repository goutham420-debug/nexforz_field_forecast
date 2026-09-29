from pathlib import Path
import pandas as pd

# Check processed first, fall back to interim if needed
processed_file = Path("data/processed/02_cleaned.parquet")
interim_file = Path("data/interim/02_cleaned.parquet")

if processed_file.exists():
    df = pd.read_parquet(processed_file)
elif interim_file.exists():
    df = pd.read_parquet(interim_file)
else:
    raise FileNotFoundError("Could not find 02_cleaned.parquet in data/processed/ or data/interim/! Run python src/clean.py first.")

summary = df[
    ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]
].describe().T
summary["cv"] = summary["std"] / summary["mean"]

report = []
report.append("# Polyhouse Data Quality Report\n")
report.append(f"Rows: {len(df)}\n")
report.append(
    f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n"
)
report.append(summary.round(2).to_markdown())

Path("reports").mkdir(exist_ok=True)
Path("reports/data_quality.md").write_text(
    "\n".join(report), encoding="utf-8"
)
print("Generated reports/data_quality.md successfully.")
print("\nSummary Statistics Table:\n")
print(summary.round(2))