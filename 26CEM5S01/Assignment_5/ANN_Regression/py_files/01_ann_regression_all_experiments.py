
from pathlib import Path
from itertools import product
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS AND EXPERIMENT SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_FILE = SCRIPT_DIR.parent / "Dataset" / "Student_Performance.csv"

OUTPUT_DIR = SCRIPT_DIR.parent / "output"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Performance Index"

RANDOM_STATE = 42
NEURONS_PER_LAYER = 16
MAX_ITERATIONS = 1000

HIDDEN_LAYER_DEPTHS = range(1, 8)

ACTIVATIONS = [
    "relu",
    "logistic",
    "tanh"
]

LEARNING_RATES = [
    0.001,
    0.005,
    0.01
]


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading Student Performance dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Check the Dataset folder and CSV filename."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found.\n"
        f"Available columns: {list(df.columns)}"
    )

if df[TARGET].isnull().any():
    raise ValueError("Target column contains missing values.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("Categorical features:", categorical_features)


# ============================================================
# 4. 80:20 TRAIN-TEST SPLIT
# ============================================================

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

# Internal validation split from the 80% training portion
X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool,
    y_train_pool,
    test_size=0.15,
    random_state=RANDOM_STATE
)

print("\nDATA SPLIT")
print("--------------------------------")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print(
    f"Training percentage: {len(X_train) / len(df) * 100:.2f}%"
)

print(
    f"Validation percentage: {len(X_val) / len(df) * 100:.2f}%"
)

print(
    f"Testing percentage: {len(X_test) / len(df) * 100:.2f}%"
)


# ============================================================
# 5. PREPROCESSING
# ============================================================

def create_preprocessor():

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# Fit preprocessing only on the training subset
preprocessor = create_preprocessor()

X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)

if hasattr(X_train_scaled, "toarray"):
    X_train_scaled = X_train_scaled.toarray()

if hasattr(X_val_scaled, "toarray"):
    X_val_scaled = X_val_scaled.toarray()

# Scale target using training subset only
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    np.asarray(y_train).reshape(-1, 1)
).ravel()

y_val_actual = np.asarray(y_val)


# ============================================================
# 6. GENERATE 63 ANN CONFIGURATIONS
# ============================================================

configurations = list(
    product(
        HIDDEN_LAYER_DEPTHS,
        ACTIVATIONS,
        LEARNING_RATES
    )
)

print("\nTotal ANN experiments:", len(configurations))


# ============================================================
# 7. RUN ALL 63 EXPERIMENTS
# ============================================================

experiment_results = []

best_validation_rmse = float("inf")
best_configuration = None

print("\nStarting ANN experiments...\n")

for experiment_number, (depth, activation, learning_rate) in enumerate(
    configurations,
    start=1
):

    hidden_layers = tuple(
        [NEURONS_PER_LAYER] * depth
    )

    print(
        f"Experiment {experiment_number}/63 | "
        f"Hidden Layers: {depth} | "
        f"Activation: {activation} | "
        f"Learning Rate: {learning_rate}"
    )

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        learning_rate_init=learning_rate,
        max_iter=MAX_ITERATIONS,
        random_state=RANDOM_STATE,
        early_stopping=False
    )

    start_time = time.time()

    model.fit(
        X_train_scaled,
        y_train_scaled
    )

    training_time = time.time() - start_time

    # Validation predictions
    val_prediction_scaled = model.predict(X_val_scaled)

    val_predictions = y_scaler.inverse_transform(
        val_prediction_scaled.reshape(-1, 1)
    ).ravel()

    # Validation metrics
    val_rmse = np.sqrt(
        mean_squared_error(y_val_actual, val_predictions)
    )

    val_mae = mean_absolute_error(
        y_val_actual,
        val_predictions
    )

    val_r2 = r2_score(
        y_val_actual,
        val_predictions
    )

    # Count trainable parameters
    parameter_count = sum(
        weights.size + biases.size
        for weights, biases in zip(
            model.coefs_,
            model.intercepts_
        )
    )

    experiment_results.append({

        "Experiment": experiment_number,
        "Hidden_Layers": depth,
        "Neurons_Per_Layer": NEURONS_PER_LAYER,
        "Hidden_Layer_Structure": str(hidden_layers),
        "Activation": activation,
        "Learning_Rate": learning_rate,

        "Validation_RMSE": val_rmse,
        "Validation_MAE": val_mae,
        "Validation_R2": val_r2,

        "Training_Time_Seconds": training_time,
        "Iterations": model.n_iter_,
        "Trainable_Parameters": parameter_count

    })

    print(
        f"RMSE: {val_rmse:.4f} | "
        f"MAE: {val_mae:.4f} | "
        f"R2: {val_r2:.4f} | "
        f"Time: {training_time:.2f}s\n"
    )

    # Select best configuration using validation RMSE
    if val_rmse < best_validation_rmse:

        best_validation_rmse = val_rmse

        best_configuration = {
            "Hidden_Layers": depth,
            "Neurons_Per_Layer": NEURONS_PER_LAYER,
            "Hidden_Layer_Structure": str(hidden_layers),
            "Activation": activation,
            "Learning_Rate": learning_rate
        }


# ============================================================
# 8. SAVE ALL 63 EXPERIMENT RESULTS
# ============================================================

results_df = pd.DataFrame(experiment_results)

results_df.to_csv(
    OUTPUT_DIR / "ann_all_63_experiments.csv",
    index=False
)

ranked_df = results_df.sort_values(
    by="Validation_RMSE",
    ascending=True
).reset_index(drop=True)

ranked_df.insert(
    0,
    "Rank",
    range(1, len(ranked_df) + 1)
)

ranked_df.to_csv(
    OUTPUT_DIR / "ann_ranked_63_experiments.csv",
    index=False
)

print("\nExperiment results saved.")


# ============================================================
# 9. ACTIVATION FUNCTION COMPARISON - 3 PLOTS
# ============================================================

activation_summary = (
    results_df
    .groupby("Activation")
    .agg(
        Average_RMSE=("Validation_RMSE", "mean"),
        Average_MAE=("Validation_MAE", "mean"),
        Average_R2=("Validation_R2", "mean")
    )
    .reindex(ACTIVATIONS)
    .reset_index()
)

activation_names = {
    "relu": "ReLU",
    "logistic": "Sigmoid",
    "tanh": "Tanh"
}

activation_summary["Activation"] = (
    activation_summary["Activation"].map(activation_names)
)


def save_bar_plot(data, x, y, title, xlabel, ylabel, filename):

    plt.figure(figsize=(8, 5))

    bars = plt.bar(data[x], data[y])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.bar_label(
        bars,
        fmt="%.4f",
        padding=3
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


save_bar_plot(
    activation_summary,
    "Activation",
    "Average_RMSE",
    "Average RMSE by Activation Function",
    "Activation Function",
    "Average RMSE",
    "01_rmse_by_activation.png"
)

save_bar_plot(
    activation_summary,
    "Activation",
    "Average_MAE",
    "Average MAE by Activation Function",
    "Activation Function",
    "Average MAE",
    "02_mae_by_activation.png"
)

save_bar_plot(
    activation_summary,
    "Activation",
    "Average_R2",
    "Average R² by Activation Function",
    "Activation Function",
    "Average R²",
    "03_r2_by_activation.png"
)


# ============================================================
# 10. HIDDEN LAYER COMPARISON - 7 PLOTS
# Each figure contains 3 panels:
# RMSE, MAE and R2.
# ============================================================

hidden_summary = (
    results_df
    .groupby("Hidden_Layers")
    .agg(
        Average_RMSE=("Validation_RMSE", "mean"),
        Average_MAE=("Validation_MAE", "mean"),
        Average_R2=("Validation_R2", "mean")
    )
    .reset_index()
)

for depth in range(1, 8):

    depth_data = results_df[
        results_df["Hidden_Layers"] == depth
    ]

    metric_summary = (
        depth_data
        .groupby("Activation")
        .agg(
            RMSE=("Validation_RMSE", "mean"),
            MAE=("Validation_MAE", "mean"),
            R2=("Validation_R2", "mean")
        )
        .reindex(ACTIVATIONS)
        .reset_index()
    )

    metric_summary["Activation"] = (
        metric_summary["Activation"].map(activation_names)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    metrics = [
        ("RMSE", "Average RMSE"),
        ("MAE", "Average MAE"),
        ("R2", "Average R²")
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):

        bars = ax.bar(
            metric_summary["Activation"],
            metric_summary[metric]
        )

        ax.set_title(metric)
        ax.set_xlabel("Activation Function")
        ax.set_ylabel(ylabel)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.5
        )

        ax.bar_label(
            bars,
            fmt="%.4f",
            padding=3
        )

    fig.suptitle(
        f"Performance Metrics - {depth} Hidden Layer(s)",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{depth + 3:02d}_hidden_layer_{depth}_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. RETRAIN BEST CONFIGURATION ON FULL 80% TRAINING DATA
# ============================================================

print("\nRetraining best configuration on the full 80% training data...")

final_preprocessor = create_preprocessor()

X_train_pool_scaled = final_preprocessor.fit_transform(
    X_train_pool
)

X_test_scaled = final_preprocessor.transform(
    X_test
)

if hasattr(X_train_pool_scaled, "toarray"):
    X_train_pool_scaled = X_train_pool_scaled.toarray()

if hasattr(X_test_scaled, "toarray"):
    X_test_scaled = X_test_scaled.toarray()

final_y_scaler = StandardScaler()

y_train_pool_scaled = final_y_scaler.fit_transform(
    np.asarray(y_train_pool).reshape(-1, 1)
).ravel()

best_depth = best_configuration["Hidden_Layers"]
best_activation = best_configuration["Activation"]
best_learning_rate = best_configuration["Learning_Rate"]

final_hidden_layers = tuple(
    [NEURONS_PER_LAYER] * best_depth
)

final_model = MLPRegressor(
    hidden_layer_sizes=final_hidden_layers,
    activation=best_activation,
    solver="adam",
    learning_rate_init=best_learning_rate,
    max_iter=MAX_ITERATIONS,
    random_state=RANDOM_STATE,
    early_stopping=False
)

final_model.fit(
    X_train_pool_scaled,
    y_train_pool_scaled
)


# ============================================================
# 12. FINAL TEST PREDICTIONS AND METRICS
# ============================================================

test_prediction_scaled = final_model.predict(
    X_test_scaled
)

test_predictions = final_y_scaler.inverse_transform(
    test_prediction_scaled.reshape(-1, 1)
).ravel()

y_test_actual = np.asarray(y_test)

test_rmse = np.sqrt(
    mean_squared_error(y_test_actual, test_predictions)
)

test_mae = mean_absolute_error(
    y_test_actual,
    test_predictions
)

test_r2 = r2_score(
    y_test_actual,
    test_predictions
)

residuals = y_test_actual - test_predictions


# ============================================================
# 13. BEST MODEL TRAINING LOSS - PLOT 11
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(final_model.loss_curve_) + 1),
    final_model.loss_curve_
)

plt.title("Best ANN Model - Training Loss")
plt.xlabel("Training Iteration")
plt.ylabel("Training Loss")

plt.grid(
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "11_best_model_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. ACTUAL VS PREDICTED - PLOT 12
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test_actual,
    test_predictions,
    alpha=0.6
)

minimum = min(
    np.min(y_test_actual),
    np.min(test_predictions)
)

maximum = max(
    np.max(y_test_actual),
    np.max(test_predictions)
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Performance Index")
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")

plt.grid(
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "12_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. RESIDUAL ERROR ANALYSIS - PLOT 13
# Four diagnostic panels in one figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 10)
)

# Panel 1: Residual distribution
axes[0, 0].hist(
    residuals,
    bins=30,
    edgecolor="black"
)

axes[0, 0].axvline(
    0,
    linestyle="--"
)

axes[0, 0].set_title("Residual Distribution")
axes[0, 0].set_xlabel("Residual Error")
axes[0, 0].set_ylabel("Frequency")


# Panel 2: Residuals vs predicted
axes[0, 1].scatter(
    test_predictions,
    residuals,
    alpha=0.6
)

axes[0, 1].axhline(
    0,
    linestyle="--"
)

axes[0, 1].set_title("Residuals vs Predicted")
axes[0, 1].set_xlabel("Predicted Performance Index")
axes[0, 1].set_ylabel("Residual Error")


# Panel 3: Residuals vs actual
axes[1, 0].scatter(
    y_test_actual,
    residuals,
    alpha=0.6
)

axes[1, 0].axhline(
    0,
    linestyle="--"
)

axes[1, 0].set_title("Residuals vs Actual")
axes[1, 0].set_xlabel("Actual Performance Index")
axes[1, 0].set_ylabel("Residual Error")


# Panel 4: Residual Q-Q plot
stats.probplot(
    residuals,
    dist="norm",
    plot=axes[1, 1]
)

axes[1, 1].set_title("Residual Q-Q Plot")

for ax in axes.flat:
    ax.grid(
        linestyle="--",
        alpha=0.4
    )

fig.suptitle(
    "Residual Error Analysis - Best ANN Model",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "13_residual_error_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. GENERATE THE 7 CSV FILES
# ============================================================

# CSV 1: All 63 experiment results
results_df.to_csv(
    OUTPUT_DIR / "ann_all_63_experiments.csv",
    index=False
)


# CSV 2: Ranked experiments
ranked_df = results_df.sort_values(
    by="Validation_RMSE",
    ascending=True
).reset_index(drop=True)

ranked_df.insert(
    0,
    "Rank",
    range(1, len(ranked_df) + 1)
)

ranked_df.to_csv(
    OUTPUT_DIR / "ann_ranked_63_experiments.csv",
    index=False
)


# CSV 3: Activation function comparison
activation_summary.to_csv(
    OUTPUT_DIR / "ann_activation_comparison.csv",
    index=False
)


# CSV 4: Hidden layer comparison
hidden_summary.to_csv(
    OUTPUT_DIR / "ann_hidden_layer_comparison.csv",
    index=False
)


# CSV 5: Resource comparison
resource_summary = (
    results_df
    .groupby("Hidden_Layers")
    .agg(
        Average_Training_Time_Seconds=(
            "Training_Time_Seconds",
            "mean"
        ),
        Average_Trainable_Parameters=(
            "Trainable_Parameters",
            "mean"
        ),
        Average_Iterations=(
            "Iterations",
            "mean"
        )
    )
    .reset_index()
)

resource_summary.to_csv(
    OUTPUT_DIR / "ann_resource_comparison.csv",
    index=False
)


# CSV 6: Best configuration and final test metrics
best_validation_row = results_df.loc[
    results_df["Validation_RMSE"].idxmin()
]

best_metrics = pd.DataFrame([{
    **best_configuration,

    "Validation_RMSE": best_validation_row["Validation_RMSE"],
    "Validation_MAE": best_validation_row["Validation_MAE"],
    "Validation_R2": best_validation_row["Validation_R2"],

    "Test_RMSE": test_rmse,
    "Test_MAE": test_mae,
    "Test_R2": test_r2
}])

best_metrics.to_csv(
    OUTPUT_DIR / "ann_best_configuration_metrics.csv",
    index=False
)


# CSV 7: Actual, predicted and residual values
actual_predicted_df = pd.DataFrame({

    "Actual_Performance_Index": y_test_actual,
    "Predicted_Performance_Index": test_predictions,
    "Residual_Error": residuals

})

actual_predicted_df.to_csv(
    OUTPUT_DIR / "ann_best_actual_vs_predicted.csv",
    index=False
)


# ============================================================
# 17. FINAL RESULTS SUMMARY
# ============================================================

print("\n")
print("=" * 55)
print("ANN REGRESSION EXPERIMENTS COMPLETED")
print("=" * 55)

print("\nBEST CONFIGURATION")
print("--------------------------------")

for key, value in best_configuration.items():
    print(f"{key}: {value}")

print("\nFINAL TEST PERFORMANCE")
print("--------------------------------")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE:  {test_mae:.4f}")
print(f"R2:   {test_r2:.4f}")

print("\nGENERATED CSV FILES")
print("--------------------------------")

csv_files = sorted(OUTPUT_DIR.glob("*.csv"))

for file in csv_files:
    print(file.name)

print("\nTotal CSV files:", len(csv_files))

print("\nGENERATED PLOT FILES")
print("--------------------------------")

plot_files = sorted(PLOT_DIR.glob("*.png"))

for file in plot_files:
    print(file.name)

print("\nTotal plots:", len(plot_files))

print("\nOutput folder:", OUTPUT_DIR)
print("Plots folder:", PLOT_DIR)

print("\nAll tasks completed successfully.")

