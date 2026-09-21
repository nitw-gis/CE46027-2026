# ============================================================
# MULTIPLE LINEAR REGRESSION vs KNN REGRESSION
# Dataset: Real estate.csv
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. LOAD DATASET
file_path = "Real estate.csv"
data = pd.read_csv(file_path)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset shape:")
print(data.shape)
print("\nColumn names:")
print(data.columns.tolist())
print("\nData types:")
print(data.dtypes)
print("\nMissing values:")
print(data.isnull().sum())
print("\nDuplicate rows:")
print(data.duplicated().sum())
print("\nStatistical summary:")
print(data.describe())

# 2. FEATURES AND TARGET
X = data.drop(columns=["No", "Y house price of unit area"])
y = data["Y house price of unit area"]

print("\nFeatures used:")
print(X.columns.tolist())
print("\nTarget variable:")
print(y.name)

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. MULTIPLE LINEAR REGRESSION
print("\n" + "=" * 60)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 60)

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
y_pred_mlr = mlr_model.predict(X_test)

mae_mlr = mean_absolute_error(y_test, y_pred_mlr)
mse_mlr = mean_squared_error(y_test, y_pred_mlr)
rmse_mlr = np.sqrt(mse_mlr)
r2_mlr = r2_score(y_test, y_pred_mlr)

print("\nMLR Performance:")
print("MAE  :", mae_mlr)
print("MSE  :", mse_mlr)
print("RMSE :", rmse_mlr)
print("R2   :", r2_mlr)

print("\nMLR Coefficients:")
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": mlr_model.coef_})
print(coefficients)
print("\nMLR Intercept:")
print(mlr_model.intercept_)

# 5. KNN REGRESSION
print("\n" + "=" * 60)
print("KNN REGRESSION")
print("=" * 60)

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])

param_grid = {"knn__n_neighbors": range(1, 21)}

grid_search = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2"
)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_["knn__n_neighbors"]
print("\nBest K value:", best_k)
print("Best cross-validation R2:", grid_search.best_score_)

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=best_k))
])
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print("\nKNN Performance:")
print("MAE  :", mae_knn)
print("MSE  :", mse_knn)
print("RMSE :", rmse_knn)
print("R2   :", r2_knn)

# 6. MODEL COMPARISON
print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": ["Multiple Linear Regression", "KNN Regression"],
    "MAE": [mae_mlr, mae_knn],
    "MSE": [mse_mlr, mse_knn],
    "RMSE": [rmse_mlr, rmse_knn],
    "R2": [r2_mlr, r2_knn]
})
print("\n", comparison.to_string(index=False))

# 7. ACTUAL VS PREDICTED
results = pd.DataFrame({
    "Actual": y_test.values,
    "MLR Predicted": y_pred_mlr,
    "KNN Predicted": y_pred_knn
})
print("\nActual vs Predicted values:")
print(results.head(20).to_string(index=False))

# 8. RESULTS FOLDER
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

comparison.to_csv(os.path.join(output_dir, "Model_Performance_Comparison.csv"), index=False)
results.to_csv(os.path.join(output_dir, "Actual_vs_Predicted.csv"), index=False)

# 9. MLR ACTUAL VS PREDICTED
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_mlr)
minimum = min(y_test.min(), y_pred_mlr.min())
maximum = max(y_test.max(), y_pred_mlr.max())
plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("MLR: Actual vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "MLR_Actual_vs_Predicted.png"), dpi=300)
plt.show()

# 10. KNN ACTUAL VS PREDICTED
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_knn)
minimum = min(y_test.min(), y_pred_knn.min())
maximum = max(y_test.max(), y_pred_knn.max())
plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("KNN: Actual vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "KNN_Actual_vs_Predicted.png"), dpi=300)
plt.show()

# 11. R2 COMPARISON
models = ["MLR", "KNN"]
plt.figure(figsize=(8, 6))
plt.bar(models, [r2_mlr, r2_knn])
plt.xlabel("Regression Model")
plt.ylabel("R2 Score")
plt.title("R2 Comparison: MLR vs KNN")
plt.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "R2_Comparison.png"), dpi=300)
plt.show()

# 12. MAE COMPARISON
plt.figure(figsize=(8, 6))
plt.bar(models, [mae_mlr, mae_knn])
plt.xlabel("Regression Model")
plt.ylabel("MAE")
plt.title("MAE Comparison: MLR vs KNN")
plt.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "MAE_Comparison.png"), dpi=300)
plt.show()

# 13. RMSE COMPARISON
plt.figure(figsize=(8, 6))
plt.bar(models, [rmse_mlr, rmse_knn])
plt.xlabel("Regression Model")
plt.ylabel("RMSE")
plt.title("RMSE Comparison: MLR vs KNN")
plt.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "RMSE_Comparison.png"), dpi=300)
plt.show()

# 14. K VALUE VS CROSS-VALIDATION R2
k_values = list(range(1, 21))
cv_scores = []
for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])
    search = GridSearchCV(model, param_grid={}, cv=5, scoring="r2")
    search.fit(X_train, y_train)
    cv_scores.append(search.best_score_)

plt.figure(figsize=(8, 6))
plt.plot(k_values, cv_scores, marker="o")
plt.xlabel("Number of Neighbours (K)")
plt.ylabel("Cross-Validation R2")
plt.title("KNN: K Value vs Cross-Validation R2")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "K_vs_CV_R2.png"), dpi=300)
plt.show()

# 15. FINAL SUMMARY
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print("\nMultiple Linear Regression")
print("--------------------------")
print("MAE  :", mae_mlr)
print("MSE  :", mse_mlr)
print("RMSE :", rmse_mlr)
print("R2   :", r2_mlr)
print("\nKNN Regression")
print("--------------------------")
print("Best K :", best_k)
print("MAE    :", mae_knn)
print("MSE    :", mse_knn)
print("RMSE   :", rmse_knn)
print("R2     :", r2_knn)
print("\nResults saved in:")
print(os.path.abspath(output_dir))
print("\nAll calculations and graphs have been completed.")
