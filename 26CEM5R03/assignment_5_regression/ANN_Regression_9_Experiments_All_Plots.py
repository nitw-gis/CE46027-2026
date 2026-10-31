# =====================================================================
# ANN REGRESSION: 7 HIDDEN LAYERS, 3 ACTIVATIONS x 3 LEARNING RATES
# Dataset: Concrete Compressive Strength (Excel/CSV)
# Metrics: R2, RMSE, MAE | CPU/RAM | execution time | README.md
# Run in Spyder (F5). Select the dataset when prompted.
# =====================================================================

import os
import time
import threading
import warnings
from pathlib import Path
from datetime import datetime
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning

# --------------------------- CONFIGURATION ----------------------------

TARGET_COLUMN = "strength"

# Seven hidden layers (seven tuples = seven hidden layers)
HIDDEN_LAYERS = (128, 64, 64, 32, 32, 16, 8)

ACTIVATIONS = ["relu", "tanh", "logistic"]
LEARNING_RATES = [0.0001, 0.001, 0.01]

TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_ITERATIONS = 1500

# --------------------------- SELECT DATASET ---------------------------

root = Tk()
root.withdraw()
root.attributes("-topmost", True)
file_path = filedialog.askopenfilename(
    title="Select your concrete strength dataset",
    filetypes=[("Excel files", "*.xlsx *.xls"),
               ("CSV files", "*.csv"),
               ("All files", "*.*")]
)
root.destroy()

if not file_path:
    raise SystemExit("No dataset selected.")

file_path = Path(file_path)
if file_path.suffix.lower() in [".xlsx", ".xls"]:
    df = pd.read_excel(file_path)
elif file_path.suffix.lower() == ".csv":
    df = pd.read_csv(file_path)
else:
    raise ValueError("Select an Excel or CSV file.")

print("\nDataset:", file_path.name)
print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------- DATA PREPARATION -------------------------

# Find target column ignoring capitalization and surrounding spaces.
target_lookup = {str(c).strip().lower(): c for c in df.columns}
if TARGET_COLUMN.lower() not in target_lookup:
    raise ValueError(
        "Target column '{}' not found. Available columns: {}"
        .format(TARGET_COLUMN, df.columns.tolist())
    )
target_col = target_lookup[TARGET_COLUMN.lower()]

df = df.dropna(subset=[target_col]).copy()
y = pd.to_numeric(df[target_col], errors="coerce")
valid_rows = y.notna()
df = df.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(float)

X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all")
X = X.fillna(X.median(numeric_only=True))

# Remove constant predictors, if any.
constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)

if X.shape[1] == 0:
    raise ValueError("No usable numerical predictor columns.")
if len(X) < 20:
    raise ValueError("Dataset is too small for this experiment.")

print("\nTarget:", target_col)
print("Observations:", len(X))
print("Predictor count:", X.shape[1])
print("Predictors:", X.columns.tolist())

# NOTE: All columns other than the target are used. Verify that none
# is calculated using the target or contains information unavailable
# at prediction time. Remove any such feature to avoid target leakage.

# --------------------------- TRAIN / TEST SPLIT -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Fit scaler on training data only: avoids test-data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------- RESOURCE MONITOR -------------------------

class ResourceMonitor:
    """Samples CPU percentage and resident memory of this Python process."""
    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.running = False
        self.peak_ram_mb = 0.0
        self.max_cpu = 0.0
        self.thread = None

    def _run(self):
        while self.running:
            try:
                ram = self.proc.memory_info().rss / (1024 ** 2)
                cpu = self.proc.cpu_percent(interval=0.1)
                self.peak_ram_mb = max(self.peak_ram_mb, ram)
                self.max_cpu = max(self.max_cpu, cpu)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def start(self):
        self.peak_ram_mb = self.proc.memory_info().rss / (1024 ** 2)
        self.max_cpu = 0.0
        self.proc.cpu_percent(interval=None)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        return self.peak_ram_mb, self.max_cpu

# --------------------------- EXPERIMENTS ------------------------------

results = []
best_model = None
best_scaler = scaler
best_f1_placeholder = None
best_rmse = float("inf")
best_details = None
best_test_predictions = None
best_train_predictions = None

# Keep predictions and training history for every experiment so that
# separate plots can be generated after all nine models have finished.
all_experiment_plot_data = []

total = len(ACTIVATIONS) * len(LEARNING_RATES)
experiment_no = 0

print("\nStarting {} ANN experiments...".format(total))
print("Architecture:", HIDDEN_LAYERS)

for activation in ACTIVATIONS:
    for lr in LEARNING_RATES:
        experiment_no += 1
        name = "ANN_{}_LR_{}".format(activation, lr)
        print("\n" + "=" * 68)
        print("Experiment {}/{}: {}".format(experiment_no, total, name))

        model = MLPRegressor(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation=activation,
            solver="adam",
            learning_rate_init=lr,
            max_iter=MAX_ITERATIONS,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=RANDOM_STATE,
            batch_size="auto",
            tol=1e-5
        )

        proc = psutil.Process(os.getpid())
        ram_before = proc.memory_info().rss / (1024 ** 2)
        cpu_before = proc.cpu_times()
        monitor = ResourceMonitor()
        monitor.start()
        start = time.perf_counter()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train_scaled, y_train)

        training_time = time.perf_counter() - start
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        total_time = time.perf_counter() - start

        peak_ram, max_cpu = monitor.stop()
        cpu_after = proc.cpu_times()
        ram_after = proc.memory_info().rss / (1024 ** 2)
        cpu_seconds = (
            cpu_after.user - cpu_before.user +
            cpu_after.system - cpu_before.system
        )
        avg_cpu = (cpu_seconds / total_time * 100) if total_time else 0

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        convergence_warning = any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )

        row = {
            "Experiment": name,
            "Activation": activation,
            "Learning_Rate": lr,
            "Hidden_Layers": str(HIDDEN_LAYERS),
            "Iterations": model.n_iter_,
            "Convergence_Warning": convergence_warning,
            "Train_R2": train_r2,
            "Test_R2": test_r2,
            "Train_RMSE": train_rmse,
            "Test_RMSE": test_rmse,
            "Train_MAE": train_mae,
            "Test_MAE": test_mae,
            "Training_Time_sec": training_time,
            "Total_Execution_Time_sec": total_time,
            "CPU_Time_sec": cpu_seconds,
            "Average_CPU_Utilization_percent": avg_cpu,
            "Maximum_Sampled_CPU_percent": max_cpu,
            "RAM_Before_MB": ram_before,
            "RAM_After_MB": ram_after,
            "Peak_Sampled_RAM_MB": peak_ram
        }
        results.append(row)

        # Save the data needed for separate plots for this experiment.
        all_experiment_plot_data.append({
            "name": name,
            "activation": activation,
            "learning_rate": lr,
            "test_predictions": test_pred.copy(),
            "train_predictions": train_pred.copy(),
            "loss_curve": list(getattr(model, "loss_curve_", [])),
            "iterations": model.n_iter_,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae
        })

        print("Train R2: {:.4f} | Test R2: {:.4f}".format(train_r2, test_r2))
        print("Test RMSE: {:.4f} | Test MAE: {:.4f}".format(
            test_rmse, test_mae))
        print("Training time: {:.3f} sec | Total: {:.3f} sec".format(
            training_time, total_time))
        print("Average CPU: {:.2f}% | Max sampled CPU: {:.2f}%".format(
            avg_cpu, max_cpu))
        print("Peak sampled RAM: {:.2f} MB".format(peak_ram))
        print("Iterations:", model.n_iter_,
              "| Convergence warning:", convergence_warning)

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model = model
            best_details = row.copy()
            best_test_predictions = test_pred.copy()
            best_train_predictions = train_pred.copy()

# --------------------------- SAVE RESULTS -----------------------------

output_dir = file_path.parent / "ANN_Regression_Results"
output_dir.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results).sort_values(
    "Test_RMSE", ascending=True
)
results_csv = output_dir / "ANN_Experiment_Results.csv"
results_df.to_csv(results_csv, index=False)

# Predictions from the best model (selected by lowest test RMSE).
predictions_df = pd.DataFrame({
    "Actual": np.asarray(y_test),
    "Predicted": best_test_predictions,
    "Residual": np.asarray(y_test) - best_test_predictions
})
predictions_df.to_csv(output_dir / "Best_Model_Predictions.csv", index=False)

# --------------------------- PLOTS ------------------------------------

# Common arrays used by the plots.
actual_test = np.asarray(y_test, dtype=float)
best_pred = np.asarray(best_test_predictions, dtype=float)
best_residuals = actual_test - best_pred

# 1. Generate three separate plots for EACH of the nine experiments:
#    (a) actual vs predicted, (b) residual scatter, (c) training loss curve.
for exp in all_experiment_plot_data:
    safe_name = exp["name"].replace(".", "_")

    pred = np.asarray(exp["test_predictions"], dtype=float)
    residual = actual_test - pred

    # Actual vs predicted for this experiment.
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_test, pred, alpha=0.7)
    lo = min(float(np.min(actual_test)), float(np.min(pred)))
    hi = max(float(np.max(actual_test)), float(np.max(pred)))
    plt.plot([lo, hi], [lo, hi], "--", label="Ideal prediction")
    plt.xlabel("Actual strength")
    plt.ylabel("Predicted strength")
    plt.title("{}: Actual vs Predicted".format(exp["name"]))
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "{}_Actual_vs_Predicted.png".format(safe_name),
                dpi=300)
    plt.close()

    # Residual scatter for this experiment.
    plt.figure(figsize=(8, 6))
    plt.scatter(pred, residual, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted strength")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("{}: Residual Plot".format(exp["name"]))
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "{}_Residual_Plot.png".format(safe_name),
                dpi=300)
    plt.close()

    # Training loss curve for this experiment.
    loss_values = exp["loss_curve"]
    if loss_values:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(loss_values) + 1), loss_values)
        plt.xlabel("Iteration")
        plt.ylabel("Training loss")
        plt.title("{}: Training Loss Curve".format(exp["name"]))
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "{}_Training_Loss.png".format(safe_name),
                    dpi=300)
        plt.close()

# 2. Additional individual diagnostic plots for the best model.
# Actual vs predicted.
plt.figure(figsize=(8, 6))
plt.scatter(actual_test, best_pred, alpha=0.7)
lo = min(float(np.min(actual_test)), float(np.min(best_pred)))
hi = max(float(np.max(actual_test)), float(np.max(best_pred)))
plt.plot([lo, hi], [lo, hi], "--", label="Ideal prediction")
plt.xlabel("Actual strength")
plt.ylabel("Predicted strength")
plt.title("Best ANN: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(output_dir / "BEST_Actual_vs_Predicted.png", dpi=300)
plt.close()

# Residuals vs predicted values.
plt.figure(figsize=(8, 6))
plt.scatter(best_pred, best_residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted strength")
plt.ylabel("Residual (actual - predicted)")
plt.title("Best ANN: Residuals vs Predicted")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(output_dir / "BEST_Residuals_vs_Predicted.png", dpi=300)
plt.close()

# Residual histogram.
plt.figure(figsize=(8, 6))
plt.hist(best_residuals, bins=25, edgecolor="black")
plt.xlabel("Residual (actual - predicted)")
plt.ylabel("Frequency")
plt.title("Best ANN: Residual Distribution")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(output_dir / "BEST_Residual_Distribution.png", dpi=300)
plt.close()

# Residuals vs actual values.
plt.figure(figsize=(8, 6))
plt.scatter(actual_test, best_residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Actual strength")
plt.ylabel("Residual (actual - predicted)")
plt.title("Best ANN: Residuals vs Actual")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(output_dir / "BEST_Residuals_vs_Actual.png", dpi=300)
plt.close()

# Training loss curve of the best model.
best_plot_data = min(all_experiment_plot_data,
                     key=lambda item: item["test_rmse"])
if best_plot_data["loss_curve"]:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(best_plot_data["loss_curve"]) + 1),
             best_plot_data["loss_curve"])
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title("Best ANN: Training Loss Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "BEST_Training_Loss.png", dpi=300)
    plt.close()

# 3. Comparison plots containing all nine experiments.
plot_results = pd.DataFrame(results)
plot_results = plot_results.sort_values("Experiment")

for metric, ylabel, title, filename in [
    ("Test_R2", "Test R-squared", "All 9 Experiments: Test R² Comparison",
     "ALL_Experiments_Test_R2.png"),
    ("Test_RMSE", "Test RMSE", "All 9 Experiments: Test RMSE Comparison",
     "ALL_Experiments_Test_RMSE.png"),
    ("Test_MAE", "Test MAE", "All 9 Experiments: Test MAE Comparison",
     "ALL_Experiments_Test_MAE.png")
]:
    plt.figure(figsize=(12, 6))
    plt.bar(plot_results["Experiment"], plot_results[metric])
    plt.xlabel("ANN experiment")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()

# --------------------------- README GENERATION ------------------------

best = best_details
table_cols = [
    ("Experiment", "Experiment"),
    ("Activation", "Activation"),
    ("Learning rate", "Learning_Rate"),
    ("Test R²", "Test_R2"),
    ("Test RMSE", "Test_RMSE"),
    ("Test MAE", "Test_MAE"),
    ("Execution time (s)", "Total_Execution_Time_sec")
]
table = [
    "| " + " | ".join(a for a, b in table_cols) + " |",
    "| " + " | ".join(["---"] * len(table_cols)) + " |"
]
for _, r in results_df.iterrows():
    vals = []
    for label, col in table_cols:
        v = r[col]
        if col in ["Test_R2", "Test_RMSE", "Test_MAE"]:
            vals.append("{:.4f}".format(v))
        elif col == "Total_Execution_Time_sec":
            vals.append("{:.3f}".format(v))
        else:
            vals.append(str(v))
    table.append("| " + " | ".join(vals) + " |")
results_table = "\n".join(table)

readme = f"""# Artificial Neural Network (ANN) Regression

## 1. Project overview

This project trains and evaluates a seven-hidden-layer Artificial
Neural Network (ANN) to predict concrete compressive strength.
Nine experiments compare three activation functions and three
learning rates.

## 2. Dataset

- File: `{file_path.name}`
- Number of observations after target cleaning: {len(X)}
- Number of predictor variables: {X.shape[1]}
- Target variable: `{target_col}`
- Problem type: supervised regression
- Train/test split: 80% / 20%
- Random state: {RANDOM_STATE}

Predictor columns used:
{chr(10).join("- `" + str(c) + "`" for c in X.columns)}

**Feature leakage check:** The code uses every numeric column except
the target. Confirm that none of the predictors is calculated from
the target or would be unavailable when predicting a new sample.
Remove any such variable before relying on the results.

## 3. ANN architecture

| Layer | Neurons |
|---|---:|
| Input | {X.shape[1]} |
| Hidden 1 | 128 |
| Hidden 2 | 64 |
| Hidden 3 | 64 |
| Hidden 4 | 32 |
| Hidden 5 | 32 |
| Hidden 6 | 16 |
| Hidden 7 | 8 |
| Output | 1 |

The output is a continuous value. The ANN is implemented with
Scikit-learn `MLPRegressor` and the Adam optimizer.

## 4. Experimental design

Three activation functions:
- ReLU (`relu`)
- Hyperbolic tangent (`tanh`)
- Logistic sigmoid (`logistic`)

Three initial learning rates:
- 0.0001
- 0.001
- 0.01

This produces 3 × 3 = 9 experiments. The architecture, split,
random state, preprocessing and maximum iteration setting are held
constant across experiments.

## 5. Preprocessing

1. Load the Excel or CSV dataset.
2. Remove rows with missing target values.
3. Convert predictor columns to numeric values.
4. Remove entirely empty and constant predictor columns.
5. Fill missing predictor values with their column medians.
6. Split into training and testing subsets.
7. Standardize features using `StandardScaler`, fitted only on
   training data.

## 6. Evaluation metrics

- **R²:** coefficient of determination. Higher is generally better;
  1 is a perfect fit, 0 is equivalent to predicting the test mean,
  and a negative value indicates worse performance than that baseline.
- **RMSE:** square root of mean squared prediction error. Lower is better.
- **MAE:** mean absolute prediction error. Lower is better.

RMSE and MAE use the same unit as the target variable.

## 7. Results from all nine experiments

Experiments are sorted by test RMSE (lowest first).

{results_table}

## 8. Best model

The selected model is the one with the lowest test RMSE among the
nine experiments. This is a comparison on one held-out test split,
so it is not an unbiased final estimate after model selection.

- Experiment: {best['Experiment']}
- Activation: {best['Activation']}
- Learning rate: {best['Learning_Rate']}
- Training R²: {best['Train_R2']:.4f}
- Testing R²: {best['Test_R2']:.4f}
- Training RMSE: {best['Train_RMSE']:.4f}
- Testing RMSE: {best['Test_RMSE']:.4f}
- Training MAE: {best['Train_MAE']:.4f}
- Testing MAE: {best['Test_MAE']:.4f}
- Iterations: {best['Iterations']}
- Convergence warning: {best['Convergence_Warning']}

## 9. Computational resource utilization

The following measurements are for the selected best-RMSE experiment:

| Resource | Value |
|---|---:|
| Training time (seconds) | {best['Training_Time_sec']:.4f} |
| Total experiment time (seconds) | {best['Total_Execution_Time_sec']:.4f} |
| CPU time (seconds) | {best['CPU_Time_sec']:.4f} |
| Average sampled CPU (%) | {best['Average_CPU_Utilization_percent']:.2f} |
| Maximum sampled CPU (%) | {best['Maximum_Sampled_CPU_percent']:.2f} |
| RAM before (MB) | {best['RAM_Before_MB']:.2f} |
| RAM after (MB) | {best['RAM_After_MB']:.2f} |
| Peak sampled RAM (MB) | {best['Peak_Sampled_RAM_MB']:.2f} |

CPU and RAM are process-level measurements. CPU utilization is
sampled during model fitting; peak RAM is the maximum sampled
resident memory of this Python process. Values depend on the
computer and background activity.

## 10. Output files

- `ANN_Experiment_Results.csv`: metrics and resource measurements
  for all nine experiments.
- `Best_Model_Predictions.csv`: actual, predicted and residual values.
- For each of the nine experiments: separate actual-vs-predicted,
  residual, and training-loss plots (where loss history is available).
- `BEST_Actual_vs_Predicted.png`, `BEST_Residuals_vs_Predicted.png`,
  `BEST_Residual_Distribution.png`, `BEST_Residuals_vs_Actual.png`,
  and `BEST_Training_Loss.png`: additional diagnostics for the best model.
- `ALL_Experiments_Test_R2.png`, `ALL_Experiments_Test_RMSE.png`,
  and `ALL_Experiments_Test_MAE.png`: metric comparisons across all nine.
- `README.md`: this documentation.

## 11. Interpretation and limitations

A higher test R² and lower test RMSE/MAE indicate better predictive
performance. Compare training and test metrics to assess possible
overfitting. A large gap may indicate that the model fits training
data better than unseen data.

A seven-hidden-layer network may be complex for a dataset of this
size. Some configurations may not converge within the iteration
limit; check the convergence-warning column. For a stronger
performance estimate, use repeated cross-validation or a separate
external test dataset. Since the nine configurations are compared
using the same test set, use a validation set or nested validation
for final model selection.

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

readme_path = output_dir / "README.md"
readme_path.write_text(readme, encoding="utf-8")

print("\n" + "=" * 70)
print("ALL NINE ANN EXPERIMENTS COMPLETED")
print("=" * 70)
print(results_df[[
    "Experiment", "Test_R2", "Test_RMSE", "Test_MAE",
    "Total_Execution_Time_sec",
    "Average_CPU_Utilization_percent", "Peak_Sampled_RAM_MB"
]].to_string(index=False))

print("\nBest model:", best["Experiment"])
print("Best test R2: {:.4f}".format(best["Test_R2"]))
print("Best test RMSE: {:.4f}".format(best["Test_RMSE"]))
print("Best test MAE: {:.4f}".format(best["Test_MAE"]))
print("\nOutput folder:", output_dir)
print("README generated:", readme_path)
