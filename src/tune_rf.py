import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Ensure directories exist
Path("models").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# 1. Load train and test data
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

# Save feature list for reproducibility (Day 15 requirement)
with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
print("Saved feature columns to: models/feature_cols.json")

# 2. Time-Series CV setup
tscv = TimeSeriesSplit(n_splits=5)

# 3. Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_test_mae = mean_absolute_error(y_test, lr_pred)
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_test_r2 = r2_score(y_test, lr_pred)

# Linear CV
from sklearn.model_selection import cross_val_score
lr_cv_mae = -cross_val_score(lr, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error").mean()

# 4. Model 2: Default Random Forest
rf_default = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_default.fit(X_train, y_train)
rf_def_pred = rf_default.predict(X_test)
rf_def_test_mae = mean_absolute_error(y_test, rf_def_pred)
rf_def_test_rmse = np.sqrt(mean_squared_error(y_test, rf_def_pred))
rf_def_test_r2 = r2_score(y_test, rf_def_pred)
rf_def_cv_mae = -cross_val_score(rf_default, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error").mean()

# 5. Model 3: GridSearchCV Hyperparameter Tuning
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 2, 4],
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
rf_tuned_pred = best_rf.predict(X_test)
rf_tuned_test_mae = mean_absolute_error(y_test, rf_tuned_pred)
rf_tuned_test_rmse = np.sqrt(mean_squared_error(y_test, rf_tuned_pred))
rf_tuned_test_r2 = r2_score(y_test, rf_tuned_pred)
rf_tuned_cv_mae = -grid_search.best_score_

# 6. Print Model Comparison Table
print("\n" + "=" * 70)
print("             MODEL COMPARISON TABLE (UNTOUCHED TEST SET)")
print("=" * 70)
print(f"{'Model':<18} | {'CV MAE':<10} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test R²':<10}")
print("-" * 70)
print(f"{'Linear Baseline':<18} | {lr_cv_mae:<10.4f} | {lr_test_mae:<10.4f} | {lr_test_rmse:<10.4f} | {lr_test_r2:<10.4f}")
print(f"{'RF Default':<18} | {rf_def_cv_mae:<10.4f} | {rf_def_test_mae:<10.4f} | {rf_def_test_rmse:<10.4f} | {rf_def_test_r2:<10.4f}")
print(f"{'RF Tuned (GridSearch)':<18} | {rf_tuned_cv_mae:<10.4f} | {rf_tuned_test_mae:<10.4f} | {rf_tuned_test_rmse:<10.4f} | {rf_tuned_test_r2:<10.4f}")
print("=" * 70)
print(f"Best Hyperparameters: {grid_search.best_params_}\n")

# 7. Select and Save Champion Model
# Tuned RF achieves best CV and Test MAE
champion_model = best_rf
joblib.dump(champion_model, "models/champion.joblib")
joblib.dump(champion_model, "models/random_forest_tuned.joblib")
print("Saved champion model to: models/champion.joblib & models/random_forest_tuned.joblib")

# 8. Plot Predicted vs Actual for Champion on Test Data
plt.figure(figsize=(7, 5))
plt.scatter(y_test, rf_tuned_pred, alpha=0.65, color="#1f77b4", edgecolor="k", s=40, label="Test Predictions")
min_val = min(y_test.min(), rf_tuned_pred.min())
max_val = max(y_test.max(), rf_tuned_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal 1:1 Line")
plt.xlabel("Actual Yield (kg)")
plt.ylabel("Predicted Yield (kg)")
plt.title("Champion Model: Predicted vs. Actual Yield (Test Set)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

fig_path = "reports/figures/pred_vs_actual.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Saved figure to: {fig_path}")

# 9. Save comparison metrics to JSON
comparison_metrics = {
    "linear": {"cv_mae": round(lr_cv_mae, 4), "test_mae": round(lr_test_mae, 4), "test_rmse": round(lr_test_rmse, 4), "test_r2": round(lr_test_r2, 4)},
    "rf_default": {"cv_mae": round(rf_def_cv_mae, 4), "test_mae": round(rf_def_test_mae, 4), "test_rmse": round(rf_def_test_rmse, 4), "test_r2": round(rf_def_test_r2, 4)},
    "rf_tuned": {"cv_mae": round(rf_tuned_cv_mae, 4), "test_mae": round(rf_tuned_test_mae, 4), "test_rmse": round(rf_tuned_test_rmse, 4), "test_r2": round(rf_tuned_test_r2, 4), "best_params": grid_search.best_params_},
}
with open("reports/model_comparison.json", "w") as f:
    json.dump(comparison_metrics, f, indent=2)
print("Saved metrics to: reports/model_comparison.json")