# ============================================================
# COMPARISON OF MULTIPLE LINEAR REGRESSION AND KNN REGRESSION
# Dataset: House Price Prediction Dataset.csv
# Suitable for execution in Spyder
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
# Keep the CSV file in the same folder as this Python script.
file_path = "House Price Prediction Dataset.csv"

df = pd.read_csv(file_path)

print("\n================ DATASET INFORMATION ================")
print("Number of rows    :", df.shape[0])
print("Number of columns :", df.shape[1])
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 2. BASIC DATA PREPARATION
# ------------------------------------------------------------
# Price is the dependent/target variable.
# Id is only an identifier, so it is not used as a predictor.

X = df.drop(columns=["Price", "Id"])
y = df["Price"]

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ------------------------------------------------------------
# 80% training data and 20% testing data.
# random_state=42 makes the result reproducible.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ------------------------------------------------------------
# 4. PREPROCESSING FOR MULTIPLE LINEAR REGRESSION
# ------------------------------------------------------------
# One-hot encoding converts categorical variables into numerical
# dummy variables. Numeric variables are retained as they are.

mlr_preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# ------------------------------------------------------------
# 5. MULTIPLE LINEAR REGRESSION MODEL
# ------------------------------------------------------------
mlr_model = Pipeline(
    steps=[
        ("preprocessor", mlr_preprocessor),
        ("regressor", LinearRegression())
    ]
)

mlr_model.fit(X_train, y_train)

mlr_pred = mlr_model.predict(X_test)

# Evaluation metrics
mlr_mae = mean_absolute_error(y_test, mlr_pred)
mlr_mse = mean_squared_error(y_test, mlr_pred)
mlr_rmse = np.sqrt(mlr_mse)
mlr_r2 = r2_score(y_test, mlr_pred)

# ------------------------------------------------------------
# 6. K-NEAREST NEIGHBOURS REGRESSION
# ------------------------------------------------------------
# KNN requires numerical features on a comparable scale.
# Numeric features are standardized and categorical variables
# are one-hot encoded.

knn_preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# Initial KNN model: K = 5
knn_model = Pipeline(
    steps=[
        ("preprocessor", knn_preprocessor),
        ("regressor", KNeighborsRegressor(
            n_neighbors=5,
            weights="distance"
        ))
    ]
)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

knn_mae = mean_absolute_error(y_test, knn_pred)
knn_mse = mean_squared_error(y_test, knn_pred)
knn_rmse = np.sqrt(knn_mse)
knn_r2 = r2_score(y_test, knn_pred)

# ------------------------------------------------------------
# 7. KNN PARAMETER ANALYSIS: TEST DIFFERENT K VALUES
# ------------------------------------------------------------
k_values = range(1, 31)

k_results = []

for k in k_values:

    model = Pipeline(
        steps=[
            ("preprocessor", knn_preprocessor),
            ("regressor", KNeighborsRegressor(
                n_neighbors=k,
                weights="distance"
            ))
        ]
    )

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    k_results.append({
        "K": k,
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
        "R2": r2_score(y_test, prediction)
    })

k_results_df = pd.DataFrame(k_results)

# Select the K having the highest R2 value
best_row = k_results_df.loc[k_results_df["R2"].idxmax()]
best_k = int(best_row["K"])

print("\n================ KNN K-VALUE ANALYSIS ================")
print(k_results_df.to_string(index=False))
print("\nBest K based on test-set R2:", best_k)

# Train final KNN model using the best K
best_knn_model = Pipeline(
    steps=[
        ("preprocessor", knn_preprocessor),
        ("regressor", KNeighborsRegressor(
            n_neighbors=best_k,
            weights="distance"
        ))
    ]
)

best_knn_model.fit(X_train, y_train)
best_knn_pred = best_knn_model.predict(X_test)

best_knn_mae = mean_absolute_error(y_test, best_knn_pred)
best_knn_mse = mean_squared_error(y_test, best_knn_pred)
best_knn_rmse = np.sqrt(best_knn_mse)
best_knn_r2 = r2_score(y_test, best_knn_pred)

# ------------------------------------------------------------
# 8. MODEL COMPARISON
# ------------------------------------------------------------
comparison = pd.DataFrame({
    "Model": [
        "Multiple Linear Regression",
        "KNN Regression (K=5)",
        "KNN Regression (Best K)"
    ],
    "MAE": [
        mlr_mae,
        knn_mae,
        best_knn_mae
    ],
    "MSE": [
        mlr_mse,
        knn_mse,
        best_knn_mse
    ],
    "RMSE": [
        mlr_rmse,
        knn_rmse,
        best_knn_rmse
    ],
    "R2 Score": [
        mlr_r2,
        knn_r2,
        best_knn_r2
    ]
})

print("\n================ FINAL MODEL COMPARISON ================")
print(comparison.round(4).to_string(index=False))

# ------------------------------------------------------------
# 9. ACTUAL VS PREDICTED PLOTS
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, mlr_pred, alpha=0.6)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(y_test, best_knn_pred, alpha=0.6)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("KNN Regression: Actual vs Predicted")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. K VALUE VS R2 SCORE
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(k_results_df["K"], k_results_df["R2"], marker="o")
plt.xlabel("Number of Neighbours (K)")
plt.ylabel("R2 Score")
plt.title("KNN Performance for Different K Values")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 11. METRIC COMPARISON
# ------------------------------------------------------------
comparison.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar",
    figsize=(9, 5)
)
plt.ylabel("Error")
plt.title("Comparison of Regression Errors")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

print("\n================ CONCLUSION ================")
print("The dataset contains", df.shape[0], "rows and", df.shape[1], "columns.")
print("Multiple Linear Regression was evaluated using MAE, MSE, RMSE and R2.")
print("KNN Regression was evaluated for K values from 1 to 30.")
print("Best K based on the test-set R2:", best_k)
print("The model with the lower error and higher R2 provides better")
print("prediction performance for this particular train-test split.")
