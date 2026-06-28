"""
Task 3 - Predictive Analytics: Sales Forecasting using Time-Series Regression
Thiranex Data Analytics Internship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(0)

print("=" * 55)
print("  PREDICTIVE ANALYTICS - THIRANEX INTERNSHIP (TASK 3)")
print("=" * 55)

# ── 1. Generate 3-year monthly sales data ──────────────────────────────────
dates = pd.date_range(start='2022-01-01', periods=36, freq='MS')
trend = np.linspace(50000, 120000, 36)
seasonality = 15000 * np.sin(2 * np.pi * np.arange(36) / 12 - np.pi/2)
noise = np.random.normal(0, 5000, 36)
sales = trend + seasonality + noise

df = pd.DataFrame({'Date': dates, 'Sales': np.round(sales, 2)})
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['t'] = np.arange(len(df))  # time index
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

print(f"\nHistorical data: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Sales range: ₹{df['Sales'].min():,.0f} – ₹{df['Sales'].max():,.0f}")
print(f"Mean sales: ₹{df['Sales'].mean():,.0f}")

# ── 2. Train/Test split ───────────────────────────────────────────────────
X = df[['t', 'Month_Sin', 'Month_Cos']]
y = df['Sales']
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── 3. Polynomial Regression (degree 2) ────────────────────────────────────
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_p = poly.fit_transform(X_train)
X_test_p  = poly.transform(X_test)
lr = LinearRegression()
lr.fit(X_train_p, y_train)
y_pred_lr = lr.predict(X_test_p)

# ── 4. Gradient Boosting Regressor ────────────────────────────────────────
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                 max_depth=4, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)

# ── 5. Evaluation ──────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  MAE  : ₹{mae:,.0f}")
    print(f"  RMSE : ₹{rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")
    return r2

r2_lr  = evaluate("Polynomial Regression (degree 2)", y_test, y_pred_lr)
r2_gbr = evaluate("Gradient Boosting Regressor",      y_test, y_pred_gbr)
best_model = "GBR" if r2_gbr > r2_lr else "Poly Regression"
print(f"\nBest model: {best_model}")

# ── 6. Future Forecast (next 12 months) ────────────────────────────────────
future_dates = pd.date_range(start='2025-01-01', periods=12, freq='MS')
t_future = np.arange(36, 48)
month_future = future_dates.month
fut_df = pd.DataFrame({
    't': t_future,
    'Month_Sin': np.sin(2 * np.pi * month_future / 12),
    'Month_Cos': np.cos(2 * np.pi * month_future / 12),
})
forecast = gbr.predict(fut_df)

print("\n12-Month Forecast (Jan 2025 – Dec 2025):")
for d, f in zip(future_dates, forecast):
    print(f"  {d.strftime('%b %Y')}: ₹{f:,.0f}")

# ── 7. Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Predictive Analytics — Sales Forecasting', fontsize=16, fontweight='bold')

# 7a. Historical + actual vs predicted
ax = axes[0, 0]
ax.plot(df['Date'], df['Sales'], label='Actual', color='#378ADD', linewidth=2)
test_dates = df['Date'].iloc[split:]
ax.plot(test_dates, y_pred_gbr, label='GBR Predicted', color='#D85A30',
        linestyle='--', linewidth=2)
ax.plot(test_dates, y_pred_lr, label='Poly Reg', color='#1D9E75',
        linestyle=':', linewidth=2)
ax.set_title('Actual vs Predicted (test set)'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_ylabel('Sales (₹)'); ax.tick_params(axis='x', rotation=30)

# 7b. 12-month forecast
ax = axes[0, 1]
ax.plot(df['Date'], df['Sales'], label='Historical', color='#378ADD', linewidth=2)
ax.plot(future_dates, forecast, label='Forecast', color='#D85A30',
        linestyle='--', linewidth=2, marker='o', markersize=4)
ax.axvline(df['Date'].max(), color='#888', linestyle=':', linewidth=1, label='Forecast start')
ax.fill_between(future_dates,
                forecast * 0.92, forecast * 1.08,
                alpha=0.15, color='#D85A30', label='±8% CI')
ax.set_title('12-Month Forecast (2025)'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_ylabel('Sales (₹)'); ax.tick_params(axis='x', rotation=30)

# 7c. Residuals
residuals = y_test.values - y_pred_gbr
ax = axes[1, 0]
ax.bar(range(len(residuals)), residuals, color=['#D85A30' if r < 0 else '#1D9E75' for r in residuals])
ax.axhline(0, color='#333', linewidth=1)
ax.set_title('Residuals (GBR — test set)'); ax.set_xlabel('Test sample index')
ax.set_ylabel('Error (₹)'); ax.grid(axis='y', alpha=0.3)

# 7d. Feature importance (GBR)
ax = axes[1, 1]
feat_names = ['Time index', 'Month Sin', 'Month Cos']
importances = gbr.feature_importances_
ax.barh(feat_names, importances, color=['#378ADD', '#1D9E75', '#BA7517'])
ax.set_title('Feature Importance (GBR)'); ax.set_xlabel('Importance score')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task3_forecast.png', dpi=150, bbox_inches='tight')
print("\nChart saved → task3_forecast.png")
df.to_csv('/mnt/user-data/outputs/task3_sales_data.csv', index=False)
print("Data saved → task3_sales_data.csv")
print("\n✓ Task 3 complete!")
