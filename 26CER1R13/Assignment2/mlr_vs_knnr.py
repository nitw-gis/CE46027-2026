import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns          # ← This line is missing in your code

# 1. Load the dataset
df = pd.read_csv("scrap price.csv")

# 2. Select numerical features and target
features = [
    'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
    'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
    'horsepower', 'peakrpm', 'citympg', 'highwaympg'
]

X = df[features]
y = df['price']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale the features (very important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# Multiple Linear Regression (MLR)
# --------------------------------------------
start_time = time.time()
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 ** 2   # in MB

mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)
y_pred_mlr = mlr.predict(X_test_scaled)

mlr_time = time.time() - start_time
mem_after = process.memory_info().rss / 1024 ** 2
mlr_memory = mem_after - mem_before

# Metrics for MLR
r2_mlr = r2_score(y_test, y_pred_mlr)
rmse_mlr = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
mae_mlr = mean_absolute_error(y_test, y_pred_mlr)

# --------------------------------------------
# K-Nearest Neighbors Regression (KNNR)
# --------------------------------------------
start_time = time.time()
mem_before = process.memory_info().rss / 1024 ** 2

knnr = KNeighborsRegressor(n_neighbors=5)
knnr.fit(X_train_scaled, y_train)
y_pred_knnr = knnr.predict(X_test_scaled)

knnr_time = time.time() - start_time
mem_after = process.memory_info().rss / 1024 ** 2
knnr_memory = mem_after - mem_before

# Metrics for KNNR
r2_knnr = r2_score(y_test, y_pred_knnr)
rmse_knnr = np.sqrt(mean_squared_error(y_test, y_pred_knnr))
mae_knnr = mean_absolute_error(y_test, y_pred_knnr)

# --------------------------------------------
# Print Comparison
# --------------------------------------------
print("="*60)
print("Performance Comparison: MLR vs KNNR")
print("="*60)

print("\nMultiple Linear Regression (MLR):")
print(f"R² Score       : {r2_mlr:.4f}")
print(f"RMSE           : {rmse_mlr:.4f}")
print(f"MAE            : {mae_mlr:.4f}")
print(f"Execution Time : {mlr_time:.4f} seconds")
print(f"Memory Used    : {mlr_memory:.2f} MB")

print("\nK-Nearest Neighbors Regression (KNNR):")
print(f"R² Score       : {r2_knnr:.4f}")
print(f"RMSE           : {rmse_knnr:.4f}")
print(f"MAE            : {mae_knnr:.4f}")
print(f"Execution Time : {knnr_time:.4f} seconds")
print(f"Memory Used    : {knnr_memory:.2f} MB")

print("\n" + "="*60)
print("Summary Table")
print("="*60)
print(f"{'Model':<10} {'R²':>10} {'RMSE':>12} {'MAE':>12} {'Time (s)':>10}")
print("-"*60)
print(f"{'MLR':<10} {r2_mlr:>10.4f} {rmse_mlr:>12.4f} {mae_mlr:>12.4f} {mlr_time:>10.4f}")
print(f"{'KNNR':<10} {r2_knnr:>10.4f} {rmse_knnr:>12.4f} {mae_knnr:>12.4f} {knnr_time:>10.4f}")

# --------------------------------------------
# PLOTS
# --------------------------------------------

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# 1. Actual vs Predicted Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MLR
axes[0].scatter(y_test, y_pred_mlr, alpha=0.7, color='blue', edgecolors='k')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title(f'MLR: Actual vs Predicted\nR² = {r2_mlr:.4f}')
axes[0].grid(True)

# KNNR
axes[1].scatter(y_test, y_pred_knnr, alpha=0.7, color='green', edgecolors='k')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title(f'KNNR: Actual vs Predicted\nR² = {r2_knnr:.4f}')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()


# 2. Residuals Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

residuals_mlr = y_test - y_pred_mlr
residuals_knnr = y_test - y_pred_knnr

axes[0].scatter(y_pred_mlr, residuals_mlr, alpha=0.7, color='blue', edgecolors='k')
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted Price')
axes[0].set_ylabel('Residuals')
axes[0].set_title('MLR Residuals Plot')
axes[0].grid(True)

axes[1].scatter(y_pred_knnr, residuals_knnr, alpha=0.7, color='green', edgecolors='k')
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Predicted Price')
axes[1].set_ylabel('Residuals')
axes[1].set_title('KNNR Residuals Plot')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
plt.show()


# 3. Metrics Comparison Bar Chart
metrics = ['R²', 'RMSE', 'MAE']
mlr_values = [r2_mlr, rmse_mlr, mae_mlr]
knnr_values = [r2_knnr, rmse_knnr, mae_knnr]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, mlr_values, width, label='MLR', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, knnr_values, width, label='KNNR', color='lightgreen', edgecolor='black')

ax.set_ylabel('Score')
ax.set_title('Performance Metrics Comparison: MLR vs KNNR')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots saved successfully:")
print("1. actual_vs_predicted.png")
print("2. residuals_plot.png")
print("3. metrics_comparison.png")