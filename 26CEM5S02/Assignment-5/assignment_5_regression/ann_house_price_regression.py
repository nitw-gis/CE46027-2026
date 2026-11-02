"""
ANN Regression Experiment: House Price Prediction
Run this file in Spyder. It trains 9 ANN models:
3 activation functions x 3 learning rates, each with 7 hidden layers.

Target: House_Price
"""

from pathlib import Path
import time
import threading
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------- USER SETTINGS -------------------------
DATASET_PATH = Path(__file__).resolve().parent / "house_price.csv.csv"
TARGET_COLUMN = "House_Price"

HIDDEN_LAYERS = (128, 64, 64, 32, 32, 16, 8)  # exactly 7 hidden layers
ACTIVATION_FUNCTIONS = {
    "ReLU": "relu",
    "Tanh": "tanh",
    "Sigmoid": "logistic"  # sklearn calls sigmoid activation 'logistic'
}
LEARNING_RATES = [0.001, 0.01, 0.1]

TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_ITERATIONS = 1500
OUTPUT_DIR = Path(__file__).resolve().parent / "ANN_Results"
# -----------------------------------------------------------------


def create_readme(dataset_path, output_dir):
    """Generate a README file describing this experiment and its outputs."""
    readme_text = f"""# ANN Regression - House Price Prediction

## Objective
Build and compare Artificial Neural Network (ANN) regression models to predict
house prices using the supplied house-price dataset.

## Dataset
- File: `{dataset_path.name}`
- Target column: `{TARGET_COLUMN}`
- Input columns: all remaining numeric columns
- Expected dataset location: same folder as the Python script

## ANN configuration
- Architecture: 7 hidden layers
- Neurons per hidden layer: `{HIDDEN_LAYERS}`
- Activation functions: ReLU, Tanh, Sigmoid (called `logistic` by Scikit-learn)
- Learning rates: `{LEARNING_RATES}`
- Number of experiments: 9 (3 activations x 3 learning rates)
- Train/test split: {int((1-TEST_SIZE)*100}% training / {int(TEST_SIZE*100)}% testing
- Maximum iterations: {MAX_ITERATIONS}
- Random state: {RANDOM_STATE}

Each experiment uses the same train/test split and standardized input features.
The target values are also standardized during model fitting and converted back
to the original house-price scale before calculating the evaluation metrics.

## Evaluation metrics
- **R-squared (R²):** goodness of fit; values closer to 1 are generally better.
- **RMSE:** square root of mean squared prediction error; lower is better.
- **MAE:** average absolute prediction error; lower is better.
- **Execution time:** model fitting time in seconds.
- **CPU utilization:** average process CPU utilization sampled during fitting.
- **Peak RAM:** maximum resident memory used by the Python process during fitting.

R² is a regression performance metric; it is not classification accuracy.

## How to run in Spyder
1. Install the required packages in the environment used by Spyder:
   `pip install pandas numpy scikit-learn joblib psutil`
2. Keep `{dataset_path.name}` and this Python script in the same folder.
3. Open the Python script in Spyder.
4. Check `DATASET_PATH` near the top of the script.
5. Click **Run** (or press F5).
6. When execution finishes, open the `ANN_Results` folder.

If Spyder uses a different Python environment, install the packages into that
same environment. The script can still run without `psutil`, but CPU/RAM
monitoring will be reported as unavailable.

## Output files
The script creates the `{output_dir.name}` folder containing:
- `ann_experiment_results.csv` - metrics and resource use for all 9 models.
- `best_ann_model.joblib` - best fitted pipeline (input scaler + target scaler + ANN).
- `best_model_predictions.csv` - actual and predicted house prices for the test set.
- `experiment_summary.txt` - best configuration and a short results summary.

## Model selection
The best model is selected by the highest test R² score. RMSE and MAE are
reported alongside R² so that prediction error is also considered.

## Important notes
- The seven layers specified here are **seven hidden layers**, followed by one
  output neuron for regression.
- The selected activation function is applied to all seven hidden layers in
  each experiment. The output layer remains linear, as appropriate for
  continuous-value regression.
- A learning rate of 0.1 may not converge as well as smaller values. Compare
  the test metrics and convergence warnings rather than assuming it is best.
- Results depend on the dataset, software versions, hardware, and train/test split.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")
    return readme_path


class ResourceMonitor:
    """Sample process CPU and RAM while model fitting is in progress."""
    def __init__(self):
        self.available = False
        self.stop_event = threading.Event()
        self.cpu_samples = []
        self.ram_samples_mb = []
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
            self.available = True
        except ImportError:
            self.psutil = None
            self.process = None

    def _sample(self):
        # Prime the CPU measurement, then sample periodically.
        self.process.cpu_percent(interval=None)
        while not self.stop_event.wait(0.2):
            try:
                self.cpu_samples.append(self.process.cpu_percent(interval=None))
                self.ram_samples_mb.append(
                    self.process.memory_info().rss / (1024 ** 2)
                )
            except Exception:
                pass

    def start(self):
        if self.available:
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()

    def stop(self):
        if self.available:
            self.stop_event.set()
            self.thread.join(timeout=2)

    def summary(self):
        if not self.available:
            return np.nan, np.nan
        avg_cpu = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        peak_ram = float(max(self.ram_samples_mb)) if self.ram_samples_mb else (
            self.process.memory_info().rss / (1024 ** 2)
        )
        return avg_cpu, peak_ram


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Place house_price.csv.csv in the same folder as this script, "
            "or update DATASET_PATH."
        )

    # 1. Load and validate dataset
    data = pd.read_csv(DATASET_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    # Keep numeric columns only; this dataset contains numeric predictors.
    numeric_data = data.select_dtypes(include=[np.number]).copy()
    if TARGET_COLUMN not in numeric_data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' must be numeric.")

    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    numeric_data = numeric_data.dropna(subset=[TARGET_COLUMN])

    X = numeric_data.drop(columns=[TARGET_COLUMN])
    y = numeric_data[TARGET_COLUMN]

    # Remove rows with missing feature values for a simple, reproducible workflow.
    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if X.shape[1] == 0:
        raise ValueError("No numeric input features were found.")

    print("=" * 68)
    print("ANN REGRESSION - HOUSE PRICE PREDICTION")
    print("=" * 68)
    print(f"Dataset: {DATASET_PATH.name}")
    print(f"Rows used: {len(X)} | Input features: {X.shape[1]}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Hidden layers: {HIDDEN_LAYERS} (7 hidden layers)")
    print(f"Experiments: {len(ACTIVATION_FUNCTIONS) * len(LEARNING_RATES)}")

    # 2. Split once so every configuration is evaluated on identical test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Standardize inputs and target. Fit scalers on training data only.
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    ).ravel()

    results = []
    fitted_models = {}

    # 4. Train all 9 combinations.
    for activation_name, activation_value in ACTIVATION_FUNCTIONS.items():
        for learning_rate in LEARNING_RATES:
            print(
                f"\nTraining: Activation={activation_name}, "
                f"Learning rate={learning_rate}"
            )

            model = MLPRegressor(
                hidden_layer_sizes=HIDDEN_LAYERS,
                activation=activation_value,
                solver="adam",
                learning_rate_init=learning_rate,
                max_iter=MAX_ITERATIONS,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=RANDOM_STATE
            )

            monitor = ResourceMonitor()
            monitor.start()
            start_time = time.perf_counter()

            try:
                model.fit(X_train_scaled, y_train_scaled)
                execution_time = time.perf_counter() - start_time
                monitor.stop()
                avg_cpu, peak_ram = monitor.summary()

                # Convert predictions back to the original currency/price scale.
                prediction_scaled = model.predict(X_test_scaled)
                predictions = y_scaler.inverse_transform(
                    prediction_scaled.reshape(-1, 1)
                ).ravel()

                r2 = r2_score(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                mae = mean_absolute_error(y_test, predictions)

                row = {
                    "Activation": activation_name,
                    "Learning_Rate": learning_rate,
                    "R2": r2,
                    "RMSE": rmse,
                    "MAE": mae,
                    "Execution_Time_Seconds": execution_time,
                    "Average_Process_CPU_Percent": avg_cpu,
                    "Peak_Process_RAM_MB": peak_ram,
                    "Iterations": model.n_iter_,
                    "Converged_or_Early_Stopped": model.n_iter_ < MAX_ITERATIONS
                }
                results.append(row)
                fitted_models[(activation_name, learning_rate)] = model

                print(
                    f"R2={r2:.4f} | RMSE={rmse:,.2f} | MAE={mae:,.2f} | "
                    f"Time={execution_time:.2f}s"
                )

            except Exception as exc:
                monitor.stop()
                print(f"Experiment failed: {exc}")
                results.append({
                    "Activation": activation_name,
                    "Learning_Rate": learning_rate,
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "Execution_Time_Seconds": time.perf_counter() - start_time,
                    "Average_Process_CPU_Percent": np.nan,
                    "Peak_Process_RAM_MB": np.nan,
                    "Iterations": np.nan,
                    "Converged_or_Early_Stopped": False,
                    "Error": str(exc)
                })

    # 5. Save comparison table and select best model by test R2.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("R2", ascending=False, na_position="last")
    results_path = OUTPUT_DIR / "ann_experiment_results.csv"
    results_df.to_csv(results_path, index=False)

    successful = results_df.dropna(subset=["R2"])
    if successful.empty:
        raise RuntimeError("All ANN experiments failed. Check the error messages.")

    best_row = successful.iloc[0]
    best_key = (best_row["Activation"], float(best_row["Learning_Rate"]))
    best_model = fitted_models[best_key]

    # Save a complete prediction pipeline: feature scaler, target scaler, ANN.
    best_pipeline = {
        "model": best_model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_columns": list(X.columns),
        "target_column": TARGET_COLUMN,
        "hidden_layers": HIDDEN_LAYERS,
        "activation": best_row["Activation"],
        "learning_rate": float(best_row["Learning_Rate"])
    }
    model_path = OUTPUT_DIR / "best_ann_model.joblib"
    joblib.dump(best_pipeline, model_path)

    best_predictions_scaled = best_model.predict(X_test_scaled)
    best_predictions = y_scaler.inverse_transform(
        best_predictions_scaled.reshape(-1, 1)
    ).ravel()

    predictions_df = pd.DataFrame({
        "Actual_House_Price": y_test.to_numpy(),
        "Predicted_House_Price": best_predictions,
        "Absolute_Error": np.abs(y_test.to_numpy() - best_predictions)
    })
    predictions_path = OUTPUT_DIR / "best_model_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # 6. Create summary and README automatically.
    summary = f"""ANN REGRESSION EXPERIMENT SUMMARY
================================
Dataset: {DATASET_PATH.name}
Rows used: {len(X)}
Input features: {X.shape[1]}
Target: {TARGET_COLUMN}
Train rows: {len(X_train)}
Test rows: {len(X_test)}
Hidden layers: {HIDDEN_LAYERS}

BEST CONFIGURATION (highest test R-squared)
Activation: {best_row['Activation']}
Learning rate: {best_row['Learning_Rate']}
R-squared: {best_row['R2']:.6f}
RMSE: {best_row['RMSE']:.6f}
MAE: {best_row['MAE']:.6f}
Execution time (seconds): {best_row['Execution_Time_Seconds']:.4f}
Average process CPU (%): {best_row['Average_Process_CPU_Percent']}
Peak process RAM (MB): {best_row['Peak_Process_RAM_MB']}

All experiment results are in ann_experiment_results.csv.
"""
    (OUTPUT_DIR / "experiment_summary.txt").write_text(summary, encoding="utf-8")
    readme_path = create_readme(DATASET_PATH, OUTPUT_DIR)

    print("\n" + "=" * 68)
    print("EXPERIMENTS COMPLETED")
    print("=" * 68)
    print(results_df.to_string(index=False))
    print("\nBest model:")
    print(f"Activation: {best_row['Activation']}")
    print(f"Learning rate: {best_row['Learning_Rate']}")
    print(f"R2: {best_row['R2']:.4f}")
    print(f"RMSE: {best_row['RMSE']:,.2f}")
    print(f"MAE: {best_row['MAE']:,.2f}")
    print(f"\nResults saved in: {OUTPUT_DIR}")
    print(f"README generated at: {readme_path}")


if __name__ == "__main__":
    main()
