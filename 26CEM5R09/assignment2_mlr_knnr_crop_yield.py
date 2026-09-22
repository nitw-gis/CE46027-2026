"""
Assignment 2: Performance Comparison of Multiple Linear Regression (MLR)
and K-Nearest Neighbors Regression (KNNR)

Dataset: Crop Yield Dataset
Target: Yield_ton_per_ha

Required comparison:
- R²
- RMSE
- MAE
- Training time
- Prediction time
- Total execution time
- Process CPU time
- Memory change
- Graphical comparison

Designed to run directly in Spyder.
"""

import os
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = "crop_yield_dataset.csv"
TARGET = "Yield_ton_per_ha"
TEST_SIZE = 0.20
RANDOM_STATE = 42
K_NEIGHBORS = 5

# All generated CSV files and graphs are saved here.
OUTPUT_DIR = Path("assignment2_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FIND AND LOAD DATASET
# ============================================================

def find_dataset(filename):
    """Find the CSV beside this script or in the current working folder."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / filename,
        Path.cwd() / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # If the CSV is not found, allow the user to select it.
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Select the crop yield CSV dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()

        if selected:
            return Path(selected)
    except Exception:
        pass

    return None


data_path = find_dataset(DATA_FILE)

if data_path is None:
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}'. Put the CSV in the same folder "
        "as this Python script or select it when prompted."
    )

print("\n" + "=" * 70)
print("ASSIGNMENT 2: MLR vs KNNR - CROP YIELD DATASET")
print("=" * 70)
print(f"Dataset file: {data_path.resolve()}")

df = pd.read_csv(data_path)

# ============================================================
# 3. DATASET INSPECTION
# ============================================================

print("\n--- DATASET INFORMATION ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nColumns:")
for col in df.columns:
    print(f"  - {col} ({df[col].dtype})")

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

missing_total = int(df.isnull().sum().sum())

duplicate_count = int(df.duplicated().sum())
print(f"\nDuplicate rows: {duplicate_count}")

# Remove duplicates, if any.
if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows after duplicate removal: {len(df)}")

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found. "
        f"Available columns: {list(df.columns)}"
    )

# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================
# This is a safety step. The current supplied dataset has been
# inspected by the author, but the script handles missing values
# if they are present in another copy of the CSV.

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Remove rows where the target itself is missing.
target_missing = int(y.isnull().sum())
if target_missing > 0:
    keep = y.notnull()
    X = X.loc[keep].copy()
    y = y.loc[keep].copy()
    print(f"\nRemoved {target_missing} rows with missing target values.")

# Identify feature types.
numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\n--- TRAIN/TEST SPLIT ---")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")
print(f"Test size    : {TEST_SIZE * 100:.0f}%")
print(f"Random state : {RANDOM_STATE}")

# ============================================================
# 6. PREPROCESSING
# ============================================================

# Compatibility with both newer and older scikit-learn versions.
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numerical_features,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]),
            categorical_features,
        ),
    ],
    remainder="drop",
)

# Same preprocessing is used for both models so the comparison is fair.

mlr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

knnr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=K_NEIGHBORS))
])


# ============================================================
# 7. MODEL EVALUATION FUNCTION
# ============================================================

process = psutil.Process(os.getpid())


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train, predict and calculate accuracy/resource metrics."""

    # Resource state before training.
    memory_before = process.memory_info().rss / (1024 ** 2)
    cpu_before = process.cpu_times().user + process.cpu_times().system

    # Training time.
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    # Prediction time.
    predict_start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - predict_start

    # Resource state after prediction.
    memory_after = process.memory_info().rss / (1024 ** 2)
    cpu_after = process.cpu_times().user + process.cpu_times().system

    total_time = train_time + predict_time
    cpu_time = cpu_after - cpu_before
    memory_change = memory_after - memory_before

    # Required regression metrics.
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    result = {
        "Model": model_name,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Training_Time_s": train_time,
        "Prediction_Time_s": predict_time,
        "Total_Execution_Time_s": total_time,
        "Process_CPU_Time_s": cpu_time,
        "Memory_Before_MB": memory_before,
        "Memory_After_MB": memory_after,
        "Memory_Change_MB": memory_change,
    }

    print(f"\n--- {model_name} RESULTS ---")
    print(f"R²                         : {r2:.6f}")
    print(f"RMSE                       : {rmse:.6f}")
    print(f"MAE                        : {mae:.6f}")
    print(f"Training time (s)          : {train_time:.6f}")
    print(f"Prediction time (s)        : {predict_time:.6f}")
    print(f"Total execution time (s)   : {total_time:.6f}")
    print(f"Process CPU time (s)       : {cpu_time:.6f}")
    print(f"Memory change (MB)         : {memory_change:.6f}")
    return result, predictions


# ============================================================
# 8. RUN MLR AND KNNR
# ============================================================

mlr_result, mlr_predictions = evaluate_model(
    "Multiple Linear Regression (MLR)",
    mlr_model,
    X_train, X_test, y_train, y_test
)

knnr_result, knnr_predictions = evaluate_model(
    f"K-Nearest Neighbors Regression (K={K_NEIGHBORS})",
    knnr_model,
    X_train, X_test, y_train, y_test
)

results_df = pd.DataFrame([mlr_result, knnr_result])

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_file = OUTPUT_DIR / "model_comparison.csv"
results_df.to_csv(results_file, index=False)

predictions_df = pd.DataFrame({
    "Actual_Yield": y_test.to_numpy(),
    "MLR_Predicted_Yield": mlr_predictions,
    "KNNR_Predicted_Yield": knnr_predictions,
})
predictions_file = OUTPUT_DIR / "predictions.csv"
predictions_df.to_csv(predictions_file, index=False)

dataset_info = pd.DataFrame({
    "Item": [
        "Dataset file",
        "Rows used",
        "Columns",
        "Target",
        "Missing values before processing",
        "Duplicate rows detected",
        "Training rows",
        "Testing rows",
        "Test size",
        "Random state",
        "K for KNNR",
        "Numerical feature count",
        "Categorical feature count",
        "Python version",
        "Pandas version",
        "NumPy version",
        "Scikit-learn version",
    ],
    "Value": [
        str(data_path.resolve()),
        len(df),
        df.shape[1],
        TARGET,
        missing_total,
        duplicate_count,
        len(X_train),
        len(X_test),
        f"{TEST_SIZE * 100:.0f}%",
        RANDOM_STATE,
        K_NEIGHBORS,
        len(numerical_features),
        len(categorical_features),
        platform.python_version(),
        pd.__version__,
        np.__version__,
        __import__("sklearn").__version__,
    ],
})

dataset_info_file = OUTPUT_DIR / "dataset_information.csv"
dataset_info.to_csv(dataset_info_file, index=False)

# ============================================================
# 10. PLOTS
# ============================================================

plt.rcParams["figure.figsize"] = (8, 6)

# 10.1 Actual vs Predicted - MLR
plt.figure()
plt.scatter(y_test, mlr_predictions, alpha=0.6)
min_v = min(y_test.min(), mlr_predictions.min())
max_v = max(y_test.max(), mlr_predictions.max())
plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
plt.xlabel("Actual Yield (ton/ha)")
plt.ylabel("Predicted Yield (ton/ha)")
plt.title("MLR: Actual vs Predicted Yield")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_MLR_actual_vs_predicted.png", dpi=300)
plt.close()

# 10.2 Actual vs Predicted - KNNR
plt.figure()
plt.scatter(y_test, knnr_predictions, alpha=0.6)
min_v = min(y_test.min(), knnr_predictions.min())
max_v = max(y_test.max(), knnr_predictions.max())
plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
plt.xlabel("Actual Yield (ton/ha)")
plt.ylabel("Predicted Yield (ton/ha)")
plt.title(f"KNNR (K={K_NEIGHBORS}): Actual vs Predicted Yield")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_KNNR_actual_vs_predicted.png", dpi=300)
plt.close()

# 10.3 Residuals - MLR
mlr_residuals = y_test.to_numpy() - mlr_predictions
plt.figure()
plt.scatter(mlr_predictions, mlr_residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Yield (ton/ha)")
plt.ylabel("Residual")
plt.title("MLR Residual Plot")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_MLR_residuals.png", dpi=300)
plt.close()

# 10.4 Residuals - KNNR
knnr_residuals = y_test.to_numpy() - knnr_predictions
plt.figure()
plt.scatter(knnr_predictions, knnr_residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Yield (ton/ha)")
plt.ylabel("Residual")
plt.title(f"KNNR (K={K_NEIGHBORS}) Residual Plot")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_KNNR_residuals.png", dpi=300)
plt.close()

# 10.5 R² comparison
plt.figure()
plt.bar(["MLR", "KNNR"], [mlr_result["R2"], knnr_result["R2"]])
plt.ylabel("R²")
plt.title("R² Comparison")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_R2_comparison.png", dpi=300)
plt.close()

# 10.6 RMSE comparison
plt.figure()
plt.bar(["MLR", "KNNR"], [mlr_result["RMSE"], knnr_result["RMSE"]])
plt.ylabel("RMSE")
plt.title("RMSE Comparison")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_RMSE_comparison.png", dpi=300)
plt.close()

# 10.7 MAE comparison
plt.figure()
plt.bar(["MLR", "KNNR"], [mlr_result["MAE"], knnr_result["MAE"]])
plt.ylabel("MAE")
plt.title("MAE Comparison")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_MAE_comparison.png", dpi=300)
plt.close()

# 10.8 Execution time comparison
plt.figure()
plt.bar(
    ["MLR", "KNNR"],
    [
        mlr_result["Total_Execution_Time_s"],
        knnr_result["Total_Execution_Time_s"],
    ],
)
plt.ylabel("Time (seconds)")
plt.title("Total Execution Time Comparison")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_execution_time_comparison.png", dpi=300)
plt.close()

# 10.9 Resource comparison - memory change
plt.figure()
plt.bar(
    ["MLR", "KNNR"],
    [
        mlr_result["Memory_Change_MB"],
        knnr_result["Memory_Change_MB"],
    ],
)
plt.ylabel("Memory Change (MB)")
plt.title("Memory Change During Model Execution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_memory_change_comparison.png", dpi=300)
plt.close()

# ============================================================
# 11. PRINT FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

display_columns = [
    "Model",
    "R2",
    "RMSE",
    "MAE",
    "Training_Time_s",
    "Prediction_Time_s",
    "Total_Execution_Time_s",
    "Process_CPU_Time_s",
    "Memory_Change_MB",
]

print(results_df[display_columns].to_string(index=False))

print("\n--- GENERATED FILES ---")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {file}")

print("\nCompleted successfully.")
print(f"Results folder: {OUTPUT_DIR.resolve()}")
print("=" * 70)
