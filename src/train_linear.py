import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure target directories exist
Path("models").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# 1. Load prepared temporal splits
train_df = pd.read_parquet("data/processed/train.parquet")
test_df = pd.read_parquet("data/processed/test.parquet")

feature_cols = [
    "temperature_c_scaled",
    "humidity_pct_scaled",
    "co2_ppm_scaled",
    "temp_humid_interaction_scaled",
]
target_col = "yield_kg"

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

# 2. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Predict on Test Set
pred_test = model.predict(X_test)
pred_train = model.predict(X_train)

# 4. Compute Evaluation Metrics
mae_test = mean_absolute_error(y_test, pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
r2_test = r2_score(y_test, pred_test)

mae_train = mean_absolute_error(y_train, pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
r2_train = r2_score(y_train, pred_train)

# Print Summary
print("=== BASELINE LINEAR REGRESSION PERFORMANCE ===")
print(f"Test MAE:  {mae_test:.3f} kg")
print(f"Test RMSE: {rmse_test:.3f} kg")
print(f"Test R²:   {r2_test:.4f}")
print(f"Train R²:  {r2_train:.4f}\n")

# Model Coefficients
feature_names = ["temperature", "humidity", "co2", "temp_humid_interaction"]
coefficients = dict(zip(feature_names, model.coef_))
coefficients["intercept"] = float(model.intercept_)

print("--- Model Coefficients ---")
for feat, coef in coefficients.items():
    print(f"  {feat}: {coef:.4f}")

# 5. Save Model Artifact
model_path = "models/linear_regression.joblib"
joblib.dump(model, model_path)
print(f"\nSaved trained model to: {model_path}")

# 6. Save Metrics to reports/linear_metrics.json
metrics_data = {
    "test_mae_kg": round(float(mae_test), 4),
    "test_rmse_kg": round(float(rmse_test), 4),
    "test_r2": round(float(r2_test), 4),
    "train_mae_kg": round(float(mae_train), 4),
    "train_rmse_kg": round(float(rmse_train), 4),
    "train_r2": round(float(r2_train), 4),
    "coefficients": {k: round(float(v), 4) for k, v in coefficients.items()},
}

metrics_path = Path("reports/linear_metrics.json")
metrics_path.write_text(json.dumps(metrics_data, indent=2))
print(f"Saved metrics to: {metrics_path}")

# 7. Generate Residual Diagnostics Plots
residuals = y_test.values - pred_test

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Residuals vs Predicted Yield
axes[0].scatter(pred_test, residuals, alpha=0.6, color="#1f77b4", edgecolor="none")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
axes[0].set_xlabel("Predicted Yield (kg)")
axes[0].set_ylabel("Residual (kg)")
axes[0].set_title("Residuals vs. Predicted Yield")
axes[0].grid(True, linestyle=":", alpha=0.6)

# Residuals vs Scaled Humidity
humidity_scaled = test_df["humidity_pct_scaled"].values
axes[1].scatter(humidity_scaled, residuals, alpha=0.6, color="#2ca02c", edgecolor="none")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.2)
axes[1].set_xlabel("Scaled Humidity")
axes[1].set_ylabel("Residual (kg)")
axes[1].set_title("Residuals vs. Scaled Humidity")
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
residual_fig_path = "reports/figures/residuals_linear.png"
plt.savefig(residual_fig_path, dpi=150)
plt.close()
print(f"Saved residual plot to: {residual_fig_path}")