# CE46027 - Assignment 2 - Gold XAUUSD
# MLR vs KNNR - Comparison

import pandas as pd
import numpy as np
import time, os, psutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Change this to your file name - you have CleanedData.xlsx
FILE = "CleanedData.xlsx" # if you renamed to gold.xlsx then change here

# Read file
try:
    if FILE.endswith(".xlsx") or FILE.endswith(".xls"):
        df = pd.read_excel(FILE)
    else:
        df = pd.read_csv(FILE)
except:
    df = pd.read_excel(FILE) # fallback

print(f"Loaded: {FILE} | Shape: {df.shape}")
print("Columns:", df.columns.tolist())

# Keep only numeric rows
df_num = df.select_dtypes(include=[np.number]).dropna()

# Auto-find target column = Close / Price
target = df_num.columns[-1] # last column default
for col in df.columns:
    if 'close' in col.lower() or 'price' in col.lower():
        if col in df_num.columns:
            target = col
        break

print(f"TARGET = {target}")

X = df_num.drop(columns=[target])
y = df_num[target]

print(f"FEATURES = {list(X.columns)}")
print(f"Data points: {len(X)}")

# Split 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

def evaluate(model, name):
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1024**2
    start_train = time.time()
    model.fit(X_train_s, y_train)
    train_time = time.time() - start_train
    mem_after = proc.memory_info().rss / 1024**2

    start_test = time.time()
    y_pred = model.predict(X_test_s)
    test_time = time.time() - start_test

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"R2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")
    print(f"Train Time = {train_time:.6f}s | Test Time = {test_time:.6f}s | Memory = {mem_after-mem_before:.2f}MB")

    return [name, r2, rmse, mae, train_time, test_time, mem_after-mem_before], y_pred

# 1. MLR
mlr_model = LinearRegression()
res_mlr, pred_mlr = evaluate(mlr_model, "Multiple Linear Regression (MLR)")

# 2. Find best K for KNNR
print("\nFinding Best K for KNNR (2 to 15)...")
best_k = 5
best_r2 = -1
for k in range(2, 16):
    knn_temp = KNeighborsRegressor(n_neighbors=k)
    knn_temp.fit(X_train_s, y_train)
    r2_temp = r2_score(y_test, knn_temp.predict(X_test_s))
    if r2_temp > best_r2:
        best_r2 = r2_temp
        best_k = k

print(f"Best K = {best_k} with R2 = {best_r2:.4f}")

knn_model = KNeighborsRegressor(n_neighbors=best_k)
res_knn, pred_knn = evaluate(knn_model, f"KNNR (K={best_k})")

# Save results
results_df = pd.DataFrame([res_mlr, res_knn],
    columns=["Model","R2_Score","RMSE","MAE","Train_Time_sec","Test_Time_sec","Memory_Used_MB"])
results_df.to_csv("comparison_results.csv", index=False)
print("\nSaved: comparison_results.csv")
print(results_df)

# Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(y_test, pred_mlr, alpha=0.5, label="MLR")
plt.scatter(y_test, pred_knn, alpha=0.5, label=f"KNNR K={best_k}")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.title("Actual vs Predicted - Gold Price")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
models = ['MLR','KNNR']
x = np.arange(3)
plt.bar(x - 0.15, [res_mlr[1], res_mlr[2], res_mlr[3]], 0.3, label='MLR')
plt.bar(x + 0.15, [res_knn[1], res_knn[2], res_knn[3]], 0.3, label='KNNR')
plt.xticks(x, ['R2','RMSE','MAE'])
plt.title("Metrics Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=300)
print("Saved: comparison_plot.png")
plt.show()

print("\n=== DONE - Upload these 3 files to GitHub ===")