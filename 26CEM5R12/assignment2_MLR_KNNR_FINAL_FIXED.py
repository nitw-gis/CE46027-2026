# ============================================================
# ASSIGNMENT 2
# PERFORMANCE COMPARISON OF MLR AND KNNR
# Dataset: FINAL_USO.csv
# Spyder-ready version
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# 1. FIND THE CSV FILE
# ============================================================
# This version first checks the folder containing this script.
# If the CSV is not found, it opens a file-selection window.
# Therefore, you do NOT need to manually type the file path.

try:
    SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_FOLDER = os.getcwd()

possible_files = [
    "FINAL_USO.csv",
    "FINAL_USO(1).csv"
]

DATA_FILE = None

for filename in possible_files:
    candidate = os.path.join(SCRIPT_FOLDER, filename)
    if os.path.isfile(candidate):
        DATA_FILE = candidate
        break

# If the file is not in the script folder, ask the user to select it.
if DATA_FILE is None:

    print("\nFINAL_USO.csv was not found automatically.")
    print("Please select your FINAL_USO CSV file in the window that opens.")

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        DATA_FILE = filedialog.askopenfilename(
            title="Select FINAL_USO CSV Dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        root.destroy()

    except Exception as error:
        print("\nCould not open the file-selection window.")
        print("Error:", error)
        raise FileNotFoundError(
            "\nPlease put FINAL_USO.csv in the same folder as this script."
        )

if not DATA_FILE:
    raise FileNotFoundError(
        "\nNo CSV file was selected. Program stopped."
    )

print("\nDataset selected:")
print(DATA_FILE)


# ============================================================
# 2. CREATE RESULTS FOLDER
# ============================================================
RESULTS_FOLDER = os.path.join(SCRIPT_FOLDER, "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================
print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Number of rows    :", df.shape[0])
print("Number of columns :", df.shape[1])


# ============================================================
# 4. DISPLAY FIRST FIVE ROWS
# ============================================================
print("\n" + "=" * 70)
print("FIRST FIVE ROWS")
print("=" * 70)
print(df.head())


# ============================================================
# 5. DATASET INFORMATION
# ============================================================
print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)
df.info()


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================
print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing = df.isnull().sum()

print(missing)
print("\nTotal missing values:", missing.sum())


# ============================================================
# 7. CHECK DUPLICATES
# ============================================================
print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")


# ============================================================
# 8. SELECT TARGET VARIABLE
# ============================================================
TARGET = "Adj Close"

if TARGET not in df.columns:

    print("\nERROR: 'Adj Close' column was not found.")
    print("\nAvailable columns:")
    for column in df.columns:
        print(column)

    raise KeyError(
        "\nThe dataset must contain an 'Adj Close' column."
    )

print("\n" + "=" * 70)
print("TARGET VARIABLE")
print("=" * 70)

print("Target variable:", TARGET)


# ============================================================
# 9. CREATE FEATURES AND TARGET
# ============================================================
# Date is excluded because it is not a numerical predictor.
# Adj Close is the dependent/target variable.

columns_to_drop = [TARGET]

if "Date" in df.columns:
    columns_to_drop.append("Date")

X = df.drop(columns=columns_to_drop)
y = df[TARGET]


# ============================================================
# 10. KEEP ONLY NUMERICAL FEATURES
# ============================================================
X = X.select_dtypes(include=[np.number])

if X.shape[1] == 0:
    raise ValueError(
        "No numerical predictor variables were found."
    )

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print("Number of features:", X.shape[1])

print("\nFeatures used:")

for i, column in enumerate(X.columns, 1):
    print(i, ".", column)


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Total samples    :", len(X))
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))
print("Training data    : 80%")
print("Testing data     : 20%")


# ============================================================
# 12. DEFINE MODELS
# ============================================================
models = {
    "Multiple Linear Regression": LinearRegression(),
    "KNN Regression": KNeighborsRegressor(
        n_neighbors=5
    )
}


# ============================================================
# 13. OPTIONAL MEMORY MONITORING
# ============================================================
try:
    import psutil
    process = psutil.Process(os.getpid())
    PSUTIL_AVAILABLE = True
except ImportError:
    process = None
    PSUTIL_AVAILABLE = False

    print("\nNOTE:")
    print("psutil is not installed.")
    print("The program will continue, but memory usage will")
    print("be shown as N/A.")


# ============================================================
# 14. RESULT STORAGE
# ============================================================
results = {}
predictions = {}


# ============================================================
# 15. TRAIN AND TEST MODELS
# ============================================================
for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    # Impute missing values, standardize features,
    # then apply the regression model.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    if PSUTIL_AVAILABLE:
        memory_before = (
            process.memory_info().rss / (1024 ** 2)
        )
    else:
        memory_before = np.nan

    start_time = time.perf_counter()
    cpu_start = time.process_time()

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    end_time = time.perf_counter()
    cpu_end = time.process_time()

    if PSUTIL_AVAILABLE:

        memory_after = (
            process.memory_info().rss / (1024 ** 2)
        )

        memory_change = (
            memory_after - memory_before
        )

    else:
        memory_change = np.nan

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    r2 = r2_score(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    execution_time = end_time - start_time
    cpu_time = cpu_end - cpu_start

    print("\nMODEL PERFORMANCE")
    print("R²             :", round(r2, 6))
    print("RMSE           :", round(rmse, 6))
    print("MAE            :", round(mae, 6))
    print(
        "Execution Time :",
        round(execution_time, 6),
        "seconds"
    )
    print(
        "CPU Time       :",
        round(cpu_time, 6),
        "seconds"
    )

    if PSUTIL_AVAILABLE:

        print(
            "Memory Change  :",
            round(memory_change, 6),
            "MB"
        )

    else:
        print("Memory Change  : N/A")

    # Store results
    results[model_name] = {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Execution_Time_Seconds": execution_time,
        "CPU_Time_Seconds": cpu_time,
        "Memory_Change_MB": memory_change
    }

    predictions[model_name] = y_pred


# ============================================================
# 16. CREATE COMPARISON TABLE
# ============================================================
results_df = pd.DataFrame(results).T.reset_index()

results_df.rename(
    columns={"index": "Model"},
    inplace=True
)


# ============================================================
# 17. DISPLAY FINAL RESULTS
# ============================================================
print("\n" + "=" * 70)
print("FINAL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================
results_file = os.path.join(
    RESULTS_FOLDER,
    "model_comparison_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\nResults saved as:")
print(results_file)


# ============================================================
# 19. SAVE DATASET INFORMATION
# ============================================================
dataset_info = pd.DataFrame({
    "Parameter": [
        "Dataset",
        "Rows",
        "Columns",
        "Target",
        "Number of Features",
        "Training Samples",
        "Testing Samples",
        "Train Percentage",
        "Test Percentage",
        "Random State",
        "K Value"
    ],

    "Value": [
        os.path.basename(DATA_FILE),
        df.shape[0],
        df.shape[1],
        TARGET,
        X.shape[1],
        X_train.shape[0],
        X_test.shape[0],
        "80%",
        "20%",
        42,
        5
    ]
})

dataset_info_file = os.path.join(
    RESULTS_FOLDER,
    "dataset_information.csv"
)

dataset_info.to_csv(
    dataset_info_file,
    index=False
)


# ============================================================
# 20. ACTUAL VS PREDICTED GRAPHS
# ============================================================
for model_name, y_pred in predictions.items():

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.7
    )

    minimum = min(
        y_test.min(),
        y_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Adj Close")
    plt.ylabel("Predicted Adj Close")

    plt.title(
        "Actual vs Predicted - " + model_name
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    if "Multiple" in model_name:
        filename = "MLR_Actual_vs_Predicted.png"
    else:
        filename = "KNNR_Actual_vs_Predicted.png"

    plt.savefig(
        os.path.join(
            RESULTS_FOLDER,
            filename
        ),
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 21. R2 COMPARISON
# ============================================================
plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.ylabel("R² Score")
plt.xlabel("Model")
plt.title("R² Comparison: MLR vs KNNR")

plt.xticks(rotation=15)
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "R2_Comparison.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 22. RMSE COMPARISON
# ============================================================
plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.ylabel("RMSE")
plt.xlabel("Model")
plt.title("RMSE Comparison: MLR vs KNNR")

plt.xticks(rotation=15)
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "RMSE_Comparison.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 23. MAE COMPARISON
# ============================================================
plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.ylabel("MAE")
plt.xlabel("Model")
plt.title("MAE Comparison: MLR vs KNNR")

plt.xticks(rotation=15)
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "MAE_Comparison.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 24. EXECUTION TIME COMPARISON
# ============================================================
plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Execution_Time_Seconds"]
)

plt.ylabel("Execution Time (seconds)")
plt.xlabel("Model")
plt.title("Execution Time Comparison")

plt.xticks(rotation=15)
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "Execution_Time_Comparison.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 25. MEMORY UTILIZATION COMPARISON
# ============================================================
if PSUTIL_AVAILABLE:

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df["Memory_Change_MB"]
    )

    plt.ylabel("Memory Change (MB)")
    plt.xlabel("Model")
    plt.title("Memory Utilization Comparison")

    plt.xticks(rotation=15)
    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "Memory_Utilization_Comparison.png"
        ),
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 26. FINAL MESSAGE
# ============================================================
print("\n" + "=" * 70)
print("ASSIGNMENT 2 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")
print("1. model_comparison_results.csv")
print("2. dataset_information.csv")
print("3. MLR_Actual_vs_Predicted.png")
print("4. KNNR_Actual_vs_Predicted.png")
print("5. R2_Comparison.png")
print("6. RMSE_Comparison.png")
print("7. MAE_Comparison.png")
print("8. Execution_Time_Comparison.png")

if PSUTIL_AVAILABLE:
    print("9. Memory_Utilization_Comparison.png")
else:
    print("9. Memory graph skipped (psutil not installed).")

print("\nAll output files are saved in:")
print(RESULTS_FOLDER)

print("\nProgram execution completed successfully!")
