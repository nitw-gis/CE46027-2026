# ==============================================================
# Assignment 2: Performance Comparison of MLR and KNNR
# Dataset: Concrete Compressive Strength
# Designed to run in Spyder
# ==============================================================

import pandas as pd
import numpy as np
import time
import tracemalloc
import psutil
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# --------------------------------------------------------------
# 1. Load the selected dataset
# --------------------------------------------------------------
# Keep this Excel file in the same folder as this Python script.
FILE_PATH = "Concrete_Compressive_Strength.xlsx"
SHEET_NAME = "Concrete Compressive Strength"

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

print("\n" + "=" * 70)
print("ASSIGNMENT 2: MLR vs KNNR")
print("=" * 70)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------------------
# 2. Basic data checking
# --------------------------------------------------------------
print("\n" + "-" * 70)
print("DATA QUALITY CHECK")
print("-" * 70)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows if any
df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)


# --------------------------------------------------------------
# 3. Define X (features) and y (target)
# --------------------------------------------------------------
# In the selected dataset, 'strength' is the target variable.
TARGET = "strength"

if TARGET not in df.columns:
    raise ValueError(
        "Target column 'strength' was not found. "
        "Check the column name in your Excel file."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\n" + "-" * 70)
print("FEATURES AND TARGET")
print("-" * 70)

print("Target variable:", TARGET)
print("Number of observations:", len(df))
print("Number of input features:", X.shape[1])
print("Features:", X.columns.tolist())


# --------------------------------------------------------------
# 4. Train-test split
# --------------------------------------------------------------
# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "-" * 70)
print("TRAIN-TEST SPLIT")
print("-" * 70)

print("Training observations:", len(X_train))
print("Testing observations :", len(X_test))


# --------------------------------------------------------------
# 5. Define models
# --------------------------------------------------------------
# MLR does not require feature scaling for ordinary least squares.
mlr = LinearRegression()

# KNN is distance-based, so StandardScaler is applied before KNN.
# k = 5 is used for this assignment.
knnr = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])


# --------------------------------------------------------------
# 6. Evaluation function
# --------------------------------------------------------------
def evaluate_model(model, model_name):

    process = psutil.Process(os.getpid())

    # Start Python memory monitoring
    tracemalloc.start()

    cpu_before = process.cpu_times()
    total_start = time.perf_counter()

    # Training
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start

    # Prediction
    prediction_start = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_time = time.perf_counter() - prediction_start

    total_time = time.perf_counter() - total_start

    # CPU time
    cpu_after = process.cpu_times()
    cpu_time = (
        (cpu_after.user + cpu_after.system)
        - (cpu_before.user + cpu_before.system)
    )

    # Peak Python memory
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak_memory / (1024 * 1024)

    # Accuracy metrics
    r2 = r2_score(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    print("R²                  :", round(r2, 6))
    print("RMSE                :", round(rmse, 6))
    print("MAE                 :", round(mae, 6))
    print("Training time (sec) :", round(training_time, 6))
    print("Prediction time(sec):", round(prediction_time, 6))
    print("Total time (sec)    :", round(total_time, 6))
    print("CPU time (sec)      :", round(cpu_time, 6))
    print("Peak memory (MB)    :", round(peak_memory_mb, 6))

    return {
        "Model": model_name,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time,
        "Total Execution Time (s)": total_time,
        "CPU Time (s)": cpu_time,
        "Peak Memory (MB)": peak_memory_mb
    }, predictions


# --------------------------------------------------------------
# 7. Evaluate MLR and KNNR
# --------------------------------------------------------------
mlr_results, mlr_predictions = evaluate_model(
    mlr,
    "Multiple Linear Regression (MLR)"
)

knnr_results, knnr_predictions = evaluate_model(
    knnr,
    "K-Nearest Neighbors Regression (KNNR, k=5)"
)


# --------------------------------------------------------------
# 8. Create final comparison table
# --------------------------------------------------------------
results = pd.DataFrame([
    mlr_results,
    knnr_results
])

print("\n" + "=" * 70)
print("FINAL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    results.round(6).to_string(index=False)
)


# --------------------------------------------------------------
# 9. Save comparison results
# --------------------------------------------------------------
results.to_csv(
    "MLR_KNNR_Comparison_Results.csv",
    index=False
)

print("\nResults saved as:")
print("MLR_KNNR_Comparison_Results.csv")


# --------------------------------------------------------------
# 10. Save actual vs predicted values
# --------------------------------------------------------------
prediction_table = pd.DataFrame({
    "Actual Strength": y_test.values,
    "MLR Predicted": mlr_predictions,
    "KNNR Predicted": knnr_predictions
})

prediction_table.to_csv(
    "Actual_vs_Predicted_Values.csv",
    index=False
)

print("Prediction values saved as:")
print("Actual_vs_Predicted_Values.csv")


# --------------------------------------------------------------
# 11. Plot: MLR actual vs predicted
# --------------------------------------------------------------
plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    mlr_predictions
)

minimum = min(y_test.min(), mlr_predictions.min())
maximum = max(y_test.max(), mlr_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Strength")
plt.ylabel("Predicted Strength")
plt.title("MLR: Actual vs Predicted Strength")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "MLR_Actual_vs_Predicted.png",
    dpi=300
)

plt.show()


# --------------------------------------------------------------
# 12. Plot: KNNR actual vs predicted
# --------------------------------------------------------------
plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    knnr_predictions
)

minimum = min(y_test.min(), knnr_predictions.min())
maximum = max(y_test.max(), knnr_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Strength")
plt.ylabel("Predicted Strength")
plt.title("KNNR: Actual vs Predicted Strength")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "KNNR_Actual_vs_Predicted.png",
    dpi=300
)

plt.show()


# --------------------------------------------------------------
# 13. Plot: R² comparison
# --------------------------------------------------------------
plt.figure(figsize=(7, 5))

plt.bar(
    ["MLR", "KNNR"],
    [mlr_results["R2"], knnr_results["R2"]]
)

plt.ylabel("R²")
plt.title("R² Comparison: MLR vs KNNR")
plt.grid(axis="y")
plt.tight_layout()

plt.savefig(
    "R2_Comparison.png",
    dpi=300
)

plt.show()


# --------------------------------------------------------------
# 14. Final interpretation
# --------------------------------------------------------------
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("\nAccuracy metrics:")
print("- Higher R² indicates a larger proportion of target variation")
print("  explained by the model.")
print("- Lower RMSE indicates smaller prediction error.")
print("- Lower MAE indicates smaller average absolute prediction error.")

print("\nComputational metrics:")
print("- Training time, prediction time and total execution time")
print("  show computational speed.")
print("- CPU time and peak Python memory give additional information")
print("  about computational resource utilization.")

print("\nNote:")
print("Execution times and memory values can vary slightly between")
print("different computers and Spyder/Python environments.")

print("\nAssignment execution completed successfully.")
print("=" * 70)
