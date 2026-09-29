from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

Path("reports/figures").mkdir(parents=True, exist_ok=True)


processed_file = Path("data/processed/02_cleaned.parquet")
interim_file = Path("data/interim/02_cleaned.parquet")

if processed_file.exists():
    df = pd.read_parquet(processed_file)
elif interim_file.exists():
    df = pd.read_parquet(interim_file)
else:
    raise FileNotFoundError("Could not find 02_cleaned.parquet in data/processed/ or data/interim/!")

features = ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]

# 1. Correlation Heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(df[features].corr(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features, rotation=45, ha="right")
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
fig.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Sensor & Yield Correlations")
plt.tight_layout()
plt.savefig("reports/figures/corr_heatmap.png", dpi=150)
plt.close()

# 2. Combined Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(df["humidity_pct"], df["yield_kg"], alpha=0.5, s=15, color="#1f77b4")
axes[0].set(xlabel="Humidity (%)", ylabel="Yield (kg)")
axes[0].set_title("Yield vs Humidity")

axes[1].scatter(df["temperature_c"], df["yield_kg"], alpha=0.5, s=15, color="#ff7f0e")
axes[1].set(xlabel="Temperature (°C)", ylabel="Yield (kg)")
axes[1].set_title("Yield vs Temperature")

axes[2].scatter(df["co2_ppm"], df["yield_kg"], alpha=0.5, s=15, color="#2ca02c")
axes[2].set(xlabel="CO2 (ppm)", ylabel="Yield (kg)")
axes[2].set_title("Yield vs CO2")

plt.tight_layout()
plt.savefig("reports/figures/scatter_yield.png", dpi=150)
plt.close()

# 3. Individual Plots to guarantee 3+ figures
plots = [
    ("humidity_pct", "Humidity", "%", "#1f77b4"),
    ("temperature_c", "Temperature", "°C", "#ff7f0e"),
    ("co2_ppm", "CO2", "ppm", "#2ca02c"),
]

for col, label, unit, color in plots:
    plt.figure(figsize=(5, 4))
    plt.scatter(df[col], df["yield_kg"], alpha=0.5, s=15, color=color)
    plt.xlabel(f"{label} ({unit})")
    plt.ylabel("Yield (kg)")
    plt.title(f"Yield (kg) vs {label} ({unit})")
    plt.tight_layout()
    plt.savefig(f"reports/figures/scatter_{col}.png", dpi=150)
    plt.close()

print("Generated plots in reports/figures/:")
print(" - corr_heatmap.png")
print(" - scatter_yield.png")
print(" - scatter_humidity_pct.png")
print(" - scatter_temperature_c.png")
print(" - scatter_co2_ppm.png")