"""
ANN Classification Experiment - Heart Disease Dataset
Run in Spyder (press F5).

Experiments:
  7 hidden layers x 3 activation functions x 3 learning rates
  3 activation functions x 3 learning rates = 9 experiments.

Target column: target (0 = no heart disease, 1 = heart disease)
"""

from pathlib import Path
import time
import threading
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------------- USER SETTINGS -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "heart.csv"
TARGET_COLUMN = "target"

# Exactly seven hidden layers
HIDDEN_LAYERS = (64, 32, 32, 16, 16, 8, 4)

ACTIVATION_FUNCTIONS = {
    "ReLU": "relu",
    "Tanh": "tanh",
    "Sigmoid": "logistic"  # Scikit-learn calls sigmoid 'logistic'
}
LEARNING_RATES = [0.001, 0.01, 0.1]

TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_ITERATIONS = 2000
OUTPUT_DIR = SCRIPT_DIR / "ANN_Classification_Results"
# -----------------------------------------------------------------


def create_readme():
    """Create project documentation automatically."""
    readme = f"""# ANN Classification - Heart Disease Dataset

## Objective
Train and compare Artificial Neural Network (ANN) binary classifiers using
the heart disease dataset.

## Dataset
- File: `{DATASET_PATH.name}`
- Number of records inspected: 303
- Number of columns: 14
- Input features: 13
- Target: `{TARGET_COLUMN}`
- Target classes: 0 (no heart disease), 1 (heart disease)
- Missing values in the supplied dataset: none

## ANN architecture and experiments
- Hidden layers: 7
- Neurons per hidden layer: `{HIDDEN_LAYERS}`
- Activation functions: ReLU, Tanh, Sigmoid (`logistic` in Scikit-learn)
- Learning rates: `{LEARNING_RATES}`
- Optimizer: Adam
- Total experiments: 9
- Train/test split: 80% / 20%, stratified by target
- Maximum iterations: {MAX_ITERATIONS}

The output layer is configured by Scikit-learn for binary classification.
The selected activation function applies to the hidden layers.

## Preprocessing
Numeric variables are imputed with the median and standardized.
Categorical-coded variables (cp, restecg, slope, thal) are imputed with the
most frequent value and one-hot encoded. Preprocessing is fitted on training
data only to avoid data leakage.

## Classification metrics
- Accuracy
- Precision
- Recall (sensitivity)
- F1-score
- Specificity
- ROC-AUC
- Confusion matrix

The positive class is `1` (heart disease). Specificity measures the fraction
of actual class-0 cases correctly predicted as class 0.

## Computational resource measurements
- Execution time: elapsed time for fitting each model, in seconds.
- Average CPU utilization: average sampled CPU use of the Python process.
- Peak RAM: maximum sampled resident memory of the Python process, in MB.

Resource measurements are approximate. Install `psutil` to enable monitoring.
CPU percentages can exceed 100% on multi-core systems.

## How to run in Spyder
1. Keep `heart.csv` and `ann_heart_disease_classification.py` in the same folder.
2. Install packages in the Python environment used by Spyder:
   `pip install pandas numpy scikit-learn joblib psutil`
3. Open `ann_heart_disease_classification.py` in Spyder.
4. Press F5 to run.
5. Check the `ANN_Classification_Results` folder.

## Output files
- `ann_classification_results.csv`: metrics for all nine experiments.
- `best_ann_classifier.joblib`: saved best preprocessing and ANN pipeline.
- `best_model_predictions.csv`: actual labels, predicted labels and probabilities.
- `best_model_confusion_matrix.csv`: confusion matrix of the best model.
- `best_model_classification_report.txt`: precision, recall and F1 by class.
- `experiment_summary.txt`: best configuration and its results.
- `README.md`: this automatically generated documentation.

## Model selection
The best model is selected by the highest test accuracy. The other metrics
are provided to show performance beyond accuracy. For a medical screening
context, recall/sensitivity and false negatives are also important.

## Important limitations
This is a small dataset (303 records). A seven-hidden-layer network may overfit
and results may vary with the train/test split. A single holdout split is not
a robust estimate of clinical performance. This exercise is for academic
machine-learning practice only and is not suitable for medical diagnosis.
"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "README.md"
    path.write_text(readme, encoding="utf-8")
    return path


class ResourceMonitor:
    """Sample CPU and resident memory use of this Python process."""
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
        self.process.cpu_percent(interval=None)  # prime measurement
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

    def get_values(self):
        if not self.available:
            return np.nan, np.nan
        avg_cpu = (
            float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        )
        peak_ram = (
            float(max(self.ram_samples_mb))
            if self.ram_samples_mb
            else self.process.memory_info().rss / (1024 ** 2)
        )
        return avg_cpu, peak_ram


def build_pipeline(X):
    """Create leakage-safe preprocessing plus ANN classifier."""
    categorical_candidates = ["cp", "restecg", "slope", "thal"]
    categorical_columns = [
        col for col in categorical_candidates if col in X.columns
    ]
    numeric_columns = [
        col for col in X.columns if col not in categorical_columns
    ]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor, numeric_columns, categorical_columns


def calculate_metrics(y_true, y_pred, y_probability):
    """Calculate binary classification metrics with class 1 as positive."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    try:
        auc = roc_auc_score(y_true, y_probability)
    except ValueError:
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "Recall_Sensitivity": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "Specificity": specificity,
        "F1_Score": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "ROC_AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATASET_PATH}. Put heart.csv in the same "
            "folder as this Python script or edit DATASET_PATH."
        )

    # 1. Load dataset and validate columns
    data = pd.read_csv(DATASET_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET_COLUMN]).copy()

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    # Ensure a binary 0/1 target
    unique_classes = sorted(pd.Series(y).unique().tolist())
    if len(unique_classes) != 2:
        raise ValueError(
            f"This script expects binary classification, but target classes "
            f"are {unique_classes}."
        )
    if set(unique_classes) != {0, 1}:
        mapping = {unique_classes[0]: 0, unique_classes[1]: 1}
        y = y.map(mapping)
        print(f"Target labels mapped as: {mapping}")

    # Convert all feature columns to numeric where possible.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Remove columns that are completely empty.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No usable input features found.")

    print("=" * 72)
    print("ANN CLASSIFICATION - HEART DISEASE DATASET")
    print("=" * 72)
    print(f"Dataset: {DATASET_PATH.name}")
    print(f"Records: {len(data)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Target classes: {sorted(pd.Series(y).unique().tolist())}")
    print(f"Hidden layers: {HIDDEN_LAYERS}")
    print("Experiments: 9")

    # 2. Split once; stratification preserves class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    results = []
    fitted_pipelines = {}

    # 3. Train all activation x learning-rate combinations.
    for activation_name, activation_value in ACTIVATION_FUNCTIONS.items():
        for learning_rate in LEARNING_RATES:
            print(
                f"\nTraining: Activation={activation_name}, "
                f"Learning rate={learning_rate}"
            )

            preprocessor, numeric_cols, categorical_cols = build_pipeline(X)

            classifier = MLPClassifier(
                hidden_layer_sizes=HIDDEN_LAYERS,
                activation=activation_value,
                solver="adam",
                learning_rate_init=learning_rate,
                max_iter=MAX_ITERATIONS,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=RANDOM_STATE
            )

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("ann", classifier)
            ])

            monitor = ResourceMonitor()
            monitor.start()
            start_time = time.perf_counter()

            try:
                pipeline.fit(X_train, y_train)
                execution_time = time.perf_counter() - start_time
                monitor.stop()
                avg_cpu, peak_ram = monitor.get_values()

                y_pred = pipeline.predict(X_test)
                y_probability = pipeline.predict_proba(X_test)[:, 1]

                metrics = calculate_metrics(y_test, y_pred, y_probability)

                row = {
                    "Activation": activation_name,
                    "Learning_Rate": learning_rate,
                    **metrics,
                    "Execution_Time_Seconds": execution_time,
                    "Average_Process_CPU_Percent": avg_cpu,
                    "Peak_Process_RAM_MB": peak_ram,
                    "Iterations": pipeline.named_steps["ann"].n_iter_,
                    "Reached_Max_Iterations": (
                        pipeline.named_steps["ann"].n_iter_ >= MAX_ITERATIONS
                    )
                }
                results.append(row)
                fitted_pipelines[(activation_name, learning_rate)] = pipeline

                print(
                    f"Accuracy={metrics['Accuracy']:.4f} | "
                    f"Precision={metrics['Precision']:.4f} | "
                    f"Recall={metrics['Recall_Sensitivity']:.4f} | "
                    f"F1={metrics['F1_Score']:.4f} | "
                    f"Time={execution_time:.2f}s"
                )

            except Exception as exc:
                monitor.stop()
                print(f"Experiment failed: {exc}")
                results.append({
                    "Activation": activation_name,
                    "Learning_Rate": learning_rate,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall_Sensitivity": np.nan,
                    "Specificity": np.nan,
                    "F1_Score": np.nan,
                    "ROC_AUC": np.nan,
                    "TN": np.nan, "FP": np.nan, "FN": np.nan, "TP": np.nan,
                    "Execution_Time_Seconds": time.perf_counter() - start_time,
                    "Average_Process_CPU_Percent": np.nan,
                    "Peak_Process_RAM_MB": np.nan,
                    "Iterations": np.nan,
                    "Reached_Max_Iterations": False,
                    "Error": str(exc)
                })

    # 4. Save experiment comparison table.
    results_df = pd.DataFrame(results).sort_values(
        "Accuracy", ascending=False, na_position="last"
    )
    results_path = OUTPUT_DIR / "ann_classification_results.csv"
    results_df.to_csv(results_path, index=False)

    successful = results_df.dropna(subset=["Accuracy"])
    if successful.empty:
        raise RuntimeError(
            "All experiments failed. Review the errors printed above."
        )

    # Select best model by test accuracy.
    best_row = successful.iloc[0]
    best_key = (
        best_row["Activation"],
        float(best_row["Learning_Rate"])
    )
    best_pipeline = fitted_pipelines[best_key]

    model_path = OUTPUT_DIR / "best_ann_classifier.joblib"
    joblib.dump(best_pipeline, model_path)

    # 5. Save predictions, probabilities and confusion matrix.
    best_predictions = best_pipeline.predict(X_test)
    best_probabilities = best_pipeline.predict_proba(X_test)[:, 1]

    predictions_df = pd.DataFrame({
        "Actual_Class": y_test.to_numpy(),
        "Predicted_Class": best_predictions,
        "Probability_Heart_Disease": best_probabilities
    })
    predictions_df.to_csv(
        OUTPUT_DIR / "best_model_predictions.csv", index=False
    )

    cm = confusion_matrix(y_test, best_predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual_0_No_Disease", "Actual_1_Disease"],
        columns=["Predicted_0_No_Disease", "Predicted_1_Disease"]
    ).to_csv(OUTPUT_DIR / "best_model_confusion_matrix.csv")

    report = classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["No heart disease (0)", "Heart disease (1)"],
        zero_division=0
    )
    (OUTPUT_DIR / "best_model_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    # 6. Save summary and generate README.
    summary = f"""ANN CLASSIFICATION EXPERIMENT SUMMARY
======================================
Dataset: {DATASET_PATH.name}
Records: {len(data)}
Input features: {X.shape[1]}
Target: {TARGET_COLUMN}
Training records: {len(X_train)}
Testing records: {len(X_test)}
Hidden layers: {HIDDEN_LAYERS}

BEST CONFIGURATION (highest test accuracy)
Activation: {best_row['Activation']}
Learning rate: {best_row['Learning_Rate']}

Accuracy: {best_row['Accuracy']:.6f}
Precision: {best_row['Precision']:.6f}
Recall / Sensitivity: {best_row['Recall_Sensitivity']:.6f}
Specificity: {best_row['Specificity']:.6f}
F1-score: {best_row['F1_Score']:.6f}
ROC-AUC: {best_row['ROC_AUC']:.6f}

Execution time (seconds): {best_row['Execution_Time_Seconds']:.4f}
Average process CPU (%): {best_row['Average_Process_CPU_Percent']}
Peak process RAM (MB): {best_row['Peak_Process_RAM_MB']}

Confusion matrix order: [[TN, FP], [FN, TP]]
TN={best_row['TN']}, FP={best_row['FP']},
FN={best_row['FN']}, TP={best_row['TP']}

All nine experiment results are in ann_classification_results.csv.

Academic note: this small dataset and one holdout split do not establish
clinical validity. This model is for educational experimentation only.
"""
    (OUTPUT_DIR / "experiment_summary.txt").write_text(
        summary, encoding="utf-8"
    )
    readme_path = create_readme()

    print("\n" + "=" * 72)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 72)
    print(results_df.to_string(index=False))
    print("\nBest model:")
    print(f"Activation: {best_row['Activation']}")
    print(f"Learning rate: {best_row['Learning_Rate']}")
    print(f"Accuracy: {best_row['Accuracy']:.4f}")
    print(f"Precision: {best_row['Precision']:.4f}")
    print(f"Recall: {best_row['Recall_Sensitivity']:.4f}")
    print(f"Specificity: {best_row['Specificity']:.4f}")
    print(f"F1-score: {best_row['F1_Score']:.4f}")
    print(f"ROC-AUC: {best_row['ROC_AUC']:.4f}")
    print(f"\nResults folder: {OUTPUT_DIR}")
    print(f"README generated: {readme_path}")


if __name__ == "__main__":
    main()
