import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Ensure target directories exist
Path("models").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# 1. Load data splits
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

# 2. Train Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Train Linear Regression for side-by-side comparison
lr = LinearRegression()
lr.fit(X_train, y_train)

# 3. Predict & Compute Hold-out Test Metrics
rf_pred_train = rf.predict(X_train)
rf_pred_test = rf.predict(X_test)
lr_pred_test = lr.predict(X_test)

# Random Forest Metrics
rf_train_mae = mean_absolute_error(y_train, rf_pred_train)
rf_test_mae = mean_absolute_error(y_test, rf_pred_test)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_pred_test))
rf_test_r2 = r2_score(y_test, rf_pred_test)

# Linear Regression Metrics
lr_test_mae = mean_absolute_error(y_test, lr_pred_test)
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_pred_test))
lr_test_r2 = r2_score(y_test, lr_pred_test)

# 4. Time-Series Cross-Validation (5 splits on train only)
tscv = TimeSeriesSplit(n_splits=5)
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")

rf_cv_mae_mean = -rf_cv_scores.mean()
rf_cv_mae_std = rf_cv_scores.std()
lr_cv_mae_mean = -lr_cv_scores.mean()
lr_cv_mae_std = lr_cv_scores.std()

# 5. Print Comparison Summary
print("\n=== MODEL COMPARISON TABLE (TEST SET) ===")
print(f"{'Metric':<15} | {'Linear Baseline':<16} | {'Random Forest':<16}")
print("-" * 52)
print(f"{'Test MAE (kg)':<15} | {lr_test_mae:<16.4f} | {rf_test_mae:<16.4f}")
print(f"{'Test RMSE (kg)':<15} | {lr_test_rmse:<16.4f} | {rf_test_rmse:<16.4f}")
print(f"{'Test R²':<15} | {lr_test_r2:<16.4f} | {rf_test_r2:<16.4f}")
print(f"{'Train MAE (kg)':<15} | {'N/A':<16} | {rf_train_mae:<16.4f}")
print(f"{'CV MAE (5-fold)':<15} | {lr_cv_mae_mean:.4f} +/- {lr_cv_mae_std:.4f} | {rf_cv_mae_mean:.4f} +/- {rf_cv_mae_std:.4f}")

# 6. Feature Importances Plot
importances = rf.feature_importances_
labels = ["Temperature", "Humidity", "CO2", "Temp x Humid Interaction"]

plt.figure(figsize=(7, 4.5))
y_pos = np.arange(len(labels))
plt.barh(y_pos, importances, align="center", color="#2b5c8f")
plt.yticks(y_pos, labels)
plt.gca().invert_yaxis()  # Highest importance at the top
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

fig_path = "reports/figures/rf_importance.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved feature importance plot to: {fig_path}")

# 7. Save Model Artifact
model_path = "models/random_forest.joblib"
joblib.dump(rf, model_path)
print(f"Saved trained Random Forest model to: {model_path}")

# 8. Save Metrics to JSON
metrics_data = {
    "random_forest": {
        "train_mae": round(float(rf_train_mae), 4),
        "test_mae": round(float(rf_test_mae), 4),
        "test_rmse": round(float(rf_test_rmse), 4),
        "test_r2": round(float(rf_test_r2), 4),
        "cv_mae_mean": round(float(rf_cv_mae_mean), 4),
        "cv_mae_std": round(float(rf_cv_mae_std), 4),
        "feature_importances": {labels[i]: round(float(importances[i]), 4) for i in range(len(labels))},
    },
    "linear_regression": {
        "test_mae": round(float(lr_test_mae), 4),
        "test_rmse": round(float(lr_test_rmse), 4),
        "test_r2": round(float(lr_test_r2), 4),
        "cv_mae_mean": round(float(lr_cv_mae_mean), 4),
        "cv_mae_std": round(float(lr_cv_mae_std), 4),
    },
}

metrics_path = Path("reports/rf_metrics.json")
metrics_path.write_text(json.dumps(metrics_data, indent=2))
print(f"Saved metrics to: {metrics_path}")