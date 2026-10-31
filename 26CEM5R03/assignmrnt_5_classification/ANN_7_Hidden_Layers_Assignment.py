# ================================================================
# ANN CLASSIFICATION ASSIGNMENT
# Dataset: Wisconsin Breast Cancer (Excel/CSV)
# Seven hidden layers; 3 activations x 3 learning rates = 9 models
# Run this script in Spyder.
# ================================================================
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl psutil

import os
import time
import threading
import warnings
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

from tkinter import Tk, filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------- CONFIGURATION ----------------------------

TARGET_COLUMN = "diagnosis"

# Exactly seven hidden layers
HIDDEN_LAYERS = (128, 64, 64, 32, 32, 16, 8)

ACTIVATION_FUNCTIONS = ["relu", "tanh", "logistic"]
LEARNING_RATES = [0.0001, 0.001, 0.01]

TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_ITERATIONS = 500

# ----------------------- SELECT DATASET ---------------------------

root = Tk()
root.withdraw()
root.attributes("-topmost", True)

file_path = filedialog.askopenfilename(
    title="Select your classification dataset",
    filetypes=[
        ("Excel files", "*.xlsx *.xls"),
        ("CSV files", "*.csv"),
        ("All files", "*.*")
    ]
)
root.destroy()

if not file_path:
    raise SystemExit("No dataset selected. Run the script again.")

if file_path.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(file_path)
else:
    df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ----------------------- DATA PREPROCESSING -----------------------

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        "Target column '{}' was not found. Available columns: {}"
        .format(TARGET_COLUMN, df.columns.tolist())
    )

# Remove rows without a target
df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Remove ID-like columns because they do not represent useful predictors.
# The uploaded breast cancer dataset contains an 'id' column.
id_columns = [
    col for col in df.columns
    if col.strip().lower() in ["id", "id_number", "patient_id"]
]

X = df.drop(columns=[TARGET_COLUMN] + id_columns)
y = df[TARGET_COLUMN].astype(str).str.strip()

# Keep numerical predictors; convert numeric-looking columns if possible.
X = X.apply(pd.to_numeric, errors="coerce")

# Remove columns that are entirely missing after conversion.
X = X.dropna(axis=1, how="all")

# Fill any remaining missing numerical values with column medians.
X = X.fillna(X.median(numeric_only=True))

if X.shape[1] == 0:
    raise ValueError("No usable numerical feature columns were found.")

# Encode class labels into integers.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

class_names = [str(c) for c in label_encoder.classes_]
print("\nTarget classes:", class_names)
print("Input features:", X.shape[1])
print("Samples:", X.shape[0])

if len(class_names) < 2:
    raise ValueError("Classification requires at least two target classes.")

# ----------------------- t-SNE VISUALIZATION ----------------------
# t-SNE is used only for exploratory visualization, not as model input.
# Scale features before calculating t-SNE.
tsne_scaler = StandardScaler()
X_scaled_for_tsne = tsne_scaler.fit_transform(X)

# Perplexity must be smaller than the number of samples.
perplexity_value = min(30, max(2, (len(X) - 1) // 3))

tsne_parameters = {
    "n_components": 2,
    "perplexity": perplexity_value,
    "learning_rate": "auto",
    "init": "pca",
    "random_state": RANDOM_STATE
}

# Compatibility with different scikit-learn versions.
if "max_iter" in inspect.signature(TSNE).parameters:
    tsne_parameters["max_iter"] = 1000
else:
    tsne_parameters["n_iter"] = 1000

print("\nGenerating 2D t-SNE plot...")
tsne = TSNE(**tsne_parameters)
X_tsne = tsne.fit_transform(X_scaled_for_tsne)

output_folder = os.path.join(
    os.path.dirname(file_path), "ANN_Assignment_Results"
)
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(10, 7))
for class_id, class_name in enumerate(class_names):
    mask = (y_encoded == class_id)
    plt.scatter(
        X_tsne[mask, 0],
        X_tsne[mask, 1],
        label=class_name,
        alpha=0.75,
        s=35
    )

plt.title("2D t-SNE Visualization of Feature Space")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Target class")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "tsne_visualization.png"),
    dpi=300
)
plt.show()

# ----------------------- TRAIN / TEST SPLIT -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# Fit scaler ONLY on training data to prevent data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining samples:", X_train_scaled.shape[0])
print("Testing samples:", X_test_scaled.shape[0])

# ----------------------- RESOURCE MONITOR -------------------------

class ResourceMonitor:
    """Samples this Python process's CPU percentage and peak RAM use."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.peak_ram_mb = 0.0
        self.max_cpu_percent = 0.0
        self.thread = None

    def _sample(self):
        while self.running:
            try:
                ram_mb = self.process.memory_info().rss / (1024 ** 2)
                cpu = self.process.cpu_percent(interval=0.1)
                self.peak_ram_mb = max(self.peak_ram_mb, ram_mb)
                self.max_cpu_percent = max(self.max_cpu_percent, cpu)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def start(self):
        self.peak_ram_mb = (
            self.process.memory_info().rss / (1024 ** 2)
        )
        self.max_cpu_percent = 0.0
        self.process.cpu_percent(interval=None)
        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        return self.peak_ram_mb, self.max_cpu_percent

# ----------------------- ANN EXPERIMENTS --------------------------

results = []
best_model = None
best_f1 = -1
best_model_details = None

total_experiments = len(ACTIVATION_FUNCTIONS) * len(LEARNING_RATES)
experiment_number = 0

print("\nStarting ANN experiments...")
print("Architecture:", HIDDEN_LAYERS)
print("Number of hidden layers:", len(HIDDEN_LAYERS))

for activation in ACTIVATION_FUNCTIONS:
    for learning_rate in LEARNING_RATES:

        experiment_number += 1
        experiment_name = "ANN_{}_LR_{}".format(
            activation, learning_rate
        )

        print("\n" + "=" * 65)
        print("Experiment {}/{}".format(
            experiment_number, total_experiments
        ))
        print("Activation:", activation)
        print("Learning rate:", learning_rate)

        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation=activation,
            solver="adam",
            learning_rate_init=learning_rate,
            max_iter=MAX_ITERATIONS,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=RANDOM_STATE,
            batch_size="auto"
        )

        monitor = ResourceMonitor()
        process = psutil.Process(os.getpid())
        ram_before_mb = process.memory_info().rss / (1024 ** 2)
        cpu_times_before = process.cpu_times()
        start_time = time.perf_counter()

        monitor.start()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        peak_ram_mb, max_cpu_percent = monitor.stop()

        end_time = time.perf_counter()
        cpu_times_after = process.cpu_times()

        execution_time = end_time - start_time
        cpu_time = (
            (cpu_times_after.user - cpu_times_before.user) +
            (cpu_times_after.system - cpu_times_before.system)
        )
        average_cpu_percent = (
            (cpu_time / execution_time) * 100
            if execution_time > 0 else 0
        )
        ram_after_mb = process.memory_info().rss / (1024 ** 2)

        # Weighted metrics account for both classes and class imbalance.
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        cm = confusion_matrix(
            y_test, y_pred, labels=np.arange(len(class_names))
        )

        results.append({
            "Experiment": experiment_name,
            "Activation": activation,
            "Learning_Rate": learning_rate,
            "Hidden_Layers": str(HIDDEN_LAYERS),
            "Iterations": model.n_iter_,
            "Accuracy": accuracy,
            "Precision_Weighted": precision,
            "Recall_Weighted": recall,
            "F1_Score_Weighted": f1,
            "Execution_Time_sec": execution_time,
            "CPU_Time_sec": cpu_time,
            "Average_CPU_Utilization_percent": average_cpu_percent,
            "Maximum_Sampled_CPU_percent": max_cpu_percent,
            "RAM_Before_MB": ram_before_mb,
            "RAM_After_MB": ram_after_mb,
            "Peak_Sampled_RAM_MB": peak_ram_mb,
            "Converged_or_Max_Iterations": (
                "Check convergence warning / n_iter"
            )
        })

        print("Accuracy  : {:.4f}".format(accuracy))
        print("Precision : {:.4f}".format(precision))
        print("Recall    : {:.4f}".format(recall))
        print("F1-score  : {:.4f}".format(f1))
        print("Iterations:", model.n_iter_)
        print("Execution time: {:.3f} seconds".format(execution_time))
        print("Average CPU utilization: {:.2f}%".format(
            average_cpu_percent
        ))
        print("Maximum sampled CPU: {:.2f}%".format(max_cpu_percent))
        print("Peak sampled RAM: {:.2f} MB".format(peak_ram_mb))

        # Save one confusion matrix for every experiment.
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title("Confusion Matrix: " + experiment_name)
        plt.xlabel("Predicted class")
        plt.ylabel("Actual class")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_folder,
                "confusion_matrix_{}.png".format(experiment_name)
            ),
            dpi=300
        )
        plt.close()

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_details = experiment_name

# ----------------------- SAVE RESULTS -----------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1_Score_Weighted", ascending=False
)

csv_path = os.path.join(output_folder, "ANN_Results.csv")
results_df.to_csv(csv_path, index=False)

print("\n" + "=" * 75)
print("FINAL COMPARISON OF ALL 9 ANN EXPERIMENTS")
print("=" * 75)
print(results_df[
    [
        "Experiment", "Accuracy", "Precision_Weighted",
        "Recall_Weighted", "F1_Score_Weighted",
        "Execution_Time_sec",
        "Average_CPU_Utilization_percent",
        "Peak_Sampled_RAM_MB"
    ]
].to_string(index=False))

print("\nBest model based on weighted F1-score:", best_model_details)
print("Best weighted F1-score: {:.4f}".format(best_f1))

# Detailed report for the best model.
best_predictions = best_model.predict(X_test_scaled)
report = classification_report(
    y_test,
    best_predictions,
    target_names=class_names,
    zero_division=0
)

report_path = os.path.join(output_folder, "Best_Model_Report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Best model: " + best_model_details + "\n\n")
    f.write("Seven hidden layers: " + str(HIDDEN_LAYERS) + "\n\n")
    f.write(report)

print("\nClassification report for the best model:")
print(report)

print("\nAll outputs saved to:")
print(output_folder)
print("\nAssignment experiments completed successfully.")
