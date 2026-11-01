
"""
DRY BEAN ANN CLASSIFICATION
63 experiments:
7 hidden-layer configurations
3 activation functions: ReLU, Sigmoid, Tanh
3 learning rates: 0.001, 0.005, 0.01

Dataset:
ANN_Classification/Dataset/Dry_Bean_Dataset.xlsx

Outputs:
ANN_Classification/output/
ANN_Classification/output/plots/
"""

from pathlib import Path
from itertools import product
import time
import gc
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND EXPERIMENT SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_FILE = PROJECT_DIR / "Dataset" / "Dry_Bean_Dataset.xlsx"

OUTPUT_DIR = PROJECT_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

RANDOM_STATE = 42

HIDDEN_LAYERS = range(1, 8)

ACTIVATIONS = [
    "ReLU",
    "Sigmoid",
    "Tanh"
]

LEARNING_RATES = [
    0.001,
    0.005,
    0.01
]

NEURONS = 16
MAX_ITER = 500


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_scores(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1_Score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }


def save_plot(fig, filename):

    fig.savefig(
        PLOTS_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 3. MAIN PROGRAM
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place Dry_Bean_Dataset.xlsx inside "
            "ANN_Classification/Dataset/."
        )

    df = pd.read_excel(DATA_FILE)

    target_column = next(
        (
            col for col in df.columns
            if str(col).strip().lower() == "class"
        ),
        None
    )

    if target_column is None:

        raise ValueError(
            "Target column 'Class' was not found."
        )

    X = df.drop(
        columns=[target_column]
    ).copy()

    for col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    if any(X[col].isna().all() for col in X.columns):

        raise ValueError(
            "At least one feature column contains no numeric values."
        )

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        df[target_column].astype(str).str.strip()
    )

    class_names = list(label_encoder.classes_)

    print("\nDATASET INFORMATION")
    print("-" * 60)

    print("Total samples:", len(df))
    print("Number of features:", X.shape[1])
    print("Number of classes:", len(class_names))
    print("Classes:", class_names)

    print(
        "Missing values:",
        int(X.isna().sum().sum())
    )

    print(
        "Duplicate rows:",
        int(df.duplicated().sum())
    )

    print("\nClass distribution:")
    print(
        df[target_column].value_counts().sort_index()
    )


    # ========================================================
    # 4. TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

    # 80% train + validation, 20% test

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Split remaining 80% into 64% train and 16% validation

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_trainval
    )

    print("\nDATA SPLIT")
    print("-" * 60)

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))


    # ========================================================
    # 5. PREPROCESSING
    # ========================================================

    preprocessor = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )

    ])

    # Fit preprocessing only on training data

    X_train_scaled = preprocessor.fit_transform(
        X_train
    )

    X_val_scaled = preprocessor.transform(
        X_val
    )


    # ========================================================
    # 6. t-SNE VISUALIZATION
    # ========================================================

    print("\nGenerating t-SNE visualization...")

    # Use a reproducible sample to keep t-SNE manageable.

    n_samples = min(
        3000,
        len(X_train_scaled)
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(X_train_scaled),
        n_samples,
        replace=False
    )

    X_sample = X_train_scaled[indices]
    y_sample = y_train[indices]

    # PCA dimensionality reduction before t-SNE

    n_components = min(
        10,
        X_sample.shape[1],
        n_samples - 1
    )

    X_pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE
    ).fit_transform(X_sample)

    perplexity = min(
        30,
        max(5, (n_samples - 1) // 3)
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE
    )

    X_tsne = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    for class_id, class_name in enumerate(class_names):

        mask = y_sample == class_id

        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=14,
            alpha=0.65,
            label=class_name
        )

    ax.set_title(
        "t-SNE Visualization of Seven Dry Bean Classes"
    )

    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

    ax.legend(
        title="Bean Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    ax.grid(alpha=0.2)

    save_plot(
        fig,
        "01_tsne_seven_bean_classes.png"
    )


    # ========================================================
    # 7. RUN ALL 63 ANN EXPERIMENTS
    # ========================================================

    results = []

    trained_models = {}

    process = psutil.Process()

    configurations = list(
        product(
            HIDDEN_LAYERS,
            ACTIVATIONS,
            LEARNING_RATES
        )
    )

    print("\nRUNNING ANN EXPERIMENTS")
    print("-" * 60)

    for experiment_number, (
        hidden_layers,
        activation,
        learning_rate
    ) in enumerate(configurations, start=1):

        # sklearn uses "logistic" internally for sigmoid.
        # The experiment name and results use "Sigmoid".

        activation_api = {
            "ReLU": "relu",
            "Sigmoid": "logistic",
            "Tanh": "tanh"
        }[activation]

        model = MLPClassifier(

            hidden_layer_sizes=(
                NEURONS,
            ) * hidden_layers,

            activation=activation_api,

            solver="adam",

            learning_rate_init=learning_rate,

            max_iter=MAX_ITER,

            random_state=RANDOM_STATE,

            early_stopping=True,

            validation_fraction=0.10,

            n_iter_no_change=25,

            tol=1e-4
        )

        # Resource measurements

        memory_before = (
            process.memory_info().rss / 1024**2
        )

        cpu_before = process.cpu_times()

        training_start = time.perf_counter()

        model.fit(
            X_train_scaled,
            y_train
        )

        training_time = (
            time.perf_counter() - training_start
        )

        cpu_after = process.cpu_times()

        cpu_time = (
            cpu_after.user
            + cpu_after.system
            - cpu_before.user
            - cpu_before.system
        )

        memory_after = (
            process.memory_info().rss / 1024**2
        )

        # Validation prediction time

        prediction_start = time.perf_counter()

        y_val_pred = model.predict(
            X_val_scaled
        )

        prediction_time = (
            time.perf_counter() - prediction_start
        )

        metrics = calculate_scores(
            y_val,
            y_val_pred
        )

        experiment_name = (
            f"ANN_{hidden_layers}Layers_"
            f"{activation}_LR_{learning_rate}"
        )

        results.append({

            "Experiment": experiment_name,

            "Hidden_Layers": hidden_layers,

            "Neurons_Per_Layer": NEURONS,

            "Activation": activation,

            "Learning_Rate": learning_rate,

            "Iterations": model.n_iter_,

            "Converged_Before_Max_Iter": (
                model.n_iter_ < MAX_ITER
            ),

            **metrics,

            "Training_Time_Seconds": training_time,

            "Prediction_Time_Seconds": prediction_time,

            "CPU_Time_Seconds": cpu_time,

            "Memory_Change_MB": (
                memory_after - memory_before
            )

        })

        trained_models[experiment_name] = model

        print(
            f"[{experiment_number:02d}/63] "
            f"{experiment_name} | "
            f"Accuracy: {metrics['Accuracy']:.4f} | "
            f"F1: {metrics['F1_Score']:.4f}"
        )

        del y_val_pred

        gc.collect()


    # ========================================================
    # 8. SAVE ALL EXPERIMENT RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    # Rank using validation metrics only

    results_df = results_df.sort_values(

        by=[
            "F1_Score",
            "Accuracy",
            "Recall",
            "Precision"
        ],

        ascending=False

    ).reset_index(drop=True)

    results_df.insert(
        0,
        "Validation_Rank",
        np.arange(1, len(results_df) + 1)
    )

    results_df.to_csv(
        OUTPUT_DIR / "ann_all_63_experiments.csv",
        index=False
    )

    results_df.to_csv(
        OUTPUT_DIR / "ann_ranked_63_experiments.csv",
        index=False
    )

    resource_columns = [

        "Experiment",
        "Hidden_Layers",
        "Neurons_Per_Layer",
        "Activation",
        "Learning_Rate",
        "Training_Time_Seconds",
        "Prediction_Time_Seconds",
        "CPU_Time_Seconds",
        "Memory_Change_MB"

    ]

    results_df[resource_columns].to_csv(
        OUTPUT_DIR / "ann_resource_comparison.csv",
        index=False
    )


    # ========================================================
    # 9. SELECT BEST MODEL USING VALIDATION RESULTS
    # ========================================================

    best_result = results_df.iloc[0]

    best_model_name = best_result["Experiment"]

    best_model = trained_models[
        best_model_name
    ]

    print("\nBEST ANN CONFIGURATION")
    print("-" * 60)

    print(best_result.to_string())


    # ========================================================
    # 10. HELD-OUT TEST EVALUATION
    # ========================================================

    # Test data is used only after selecting the best model.

    X_test_scaled = preprocessor.transform(
        X_test
    )

    test_start = time.perf_counter()

    y_test_pred = best_model.predict(
        X_test_scaled
    )

    test_prediction_time = (
        time.perf_counter() - test_start
    )

    test_metrics = calculate_scores(
        y_test,
        y_test_pred
    )

    test_metrics_df = pd.DataFrame([{

        "Best_Experiment": best_model_name,

        "Hidden_Layers": int(
            best_result["Hidden_Layers"]
        ),

        "Neurons_Per_Layer": NEURONS,

        "Activation": best_result["Activation"],

        "Learning_Rate": best_result["Learning_Rate"],

        **test_metrics,

        "Test_Prediction_Time_Seconds": test_prediction_time

    }])

    test_metrics_df.to_csv(
        OUTPUT_DIR / "best_ann_heldout_test_metrics.csv",
        index=False
    )


    # ========================================================
    # 11. PER-CLASS PERFORMANCE METRICS
    # ========================================================

    report = classification_report(

        y_test,
        y_test_pred,

        labels=np.arange(len(class_names)),

        target_names=class_names,

        output_dict=True,

        zero_division=0

    )

    per_class_results = []

    for class_name in class_names:

        per_class_results.append({

            "Class": class_name,

            "Precision": report[class_name]["precision"],

            "Recall": report[class_name]["recall"],

            "F1_Score": report[class_name]["f1-score"],

            "Support": int(
                report[class_name]["support"]
            )

        })

    pd.DataFrame(
        per_class_results
    ).to_csv(
        OUTPUT_DIR / "best_ann_per_class_metrics.csv",
        index=False
    )


    # ========================================================
    # 12. SAVE TEST PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame({

        "Test_Row_Index": X_test.index,

        "Actual_Class": label_encoder.inverse_transform(
            y_test
        ),

        "Predicted_Class": label_encoder.inverse_transform(
            y_test_pred
        ),

        "Correct": (
            y_test == y_test_pred
        )

    })

    predictions_df.to_csv(
        OUTPUT_DIR / "best_ann_test_predictions.csv",
        index=False
    )


    # ========================================================
    # 13. BEST ANN CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        y_test,
        y_test_pred,

        labels=np.arange(len(class_names))

    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=class_names,

        yticklabels=class_names,

        ax=ax

    )

    ax.set_title(
        "Best ANN - Held-out Test Confusion Matrix"
    )

    ax.set_xlabel("Predicted Class")

    ax.set_ylabel("Actual Class")

    plt.setp(
        ax.get_xticklabels(),
        rotation=35,
        ha="right"
    )

    save_plot(
        fig,
        "02_best_ann_test_confusion_matrix.png"
    )


    # ========================================================
    # 14. HIDDEN-LAYER COMPARISON PLOTS (7 PLOTS)
    # ========================================================

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]

    for depth in HIDDEN_LAYERS:

        subset = results_df[
            results_df["Hidden_Layers"] == depth
        ]

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10)
        )

        for ax, metric in zip(
            axes.ravel(),
            metrics
        ):

            sns.lineplot(

                data=subset,

                x="Learning_Rate",

                y=metric,

                hue="Activation",

                style="Activation",

                markers=True,

                dashes=False,

                hue_order=ACTIVATIONS,

                ax=ax

            )

            ax.set_title(
                metric.replace("_", " ")
            )

            ax.set_xlabel(
                "Learning Rate"
            )

            ax.set_ylabel(
                metric.replace("_", " ")
            )

            ax.set_xticks(
                LEARNING_RATES
            )

            ax.grid(alpha=0.25)

        fig.suptitle(
            f"Validation Metrics - {depth} Hidden Layer(s), "
            f"16 Neurons per Layer"
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        save_plot(
            fig,
            f"hidden_layer_{depth}_metric_comparison.png"
        )


    # ========================================================
    # 15. ACTIVATION-FUNCTION COMPARISON (4 PLOTS)
    # ========================================================

    activation_means = (

        results_df.groupby("Activation")[metrics]

        .mean()

        .reindex(ACTIVATIONS)

        .reset_index()

    )

    for metric in metrics:

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        sns.barplot(

            data=activation_means,

            x="Activation",

            y=metric,

            order=ACTIVATIONS,

            ax=ax

        )

        ax.set_title(
            f"Mean Validation {metric.replace('_', ' ')} "
            "by Activation"
        )

        ax.set_xlabel(
            "Activation Function"
        )

        ax.set_ylabel(
            f"Mean Validation {metric.replace('_', ' ')}"
        )

        ax.set_ylim(0, 1.05)

        ax.grid(
            axis="y",
            alpha=0.25
        )

        save_plot(
            fig,
            f"activation_comparison_{metric.lower()}.png"
        )


    # ========================================================
    # 16. OVERALL PERFORMANCE (3 PLOTS)
    # ========================================================

    layer_means = (

        results_df.groupby("Hidden_Layers")[
            ["Accuracy", "Recall", "F1_Score"]
        ]

        .mean()

        .reset_index()

    )

    for metric in [
        "Accuracy",
        "Recall",
        "F1_Score"
    ]:

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        sns.lineplot(

            data=layer_means,

            x="Hidden_Layers",

            y=metric,

            marker="o",

            ax=ax

        )

        ax.set_title(
            f"Overall Validation {metric.replace('_', ' ')} "
            "Across Hidden Layers"
        )

        ax.set_xlabel(
            "Number of Hidden Layers"
        )

        ax.set_ylabel(
            f"Mean Validation {metric.replace('_', ' ')}"
        )

        ax.set_xticks(
            list(HIDDEN_LAYERS)
        )

        ax.set_ylim(0, 1.05)

        ax.grid(alpha=0.25)

        save_plot(
            fig,
            f"overall_hidden_layer_{metric.lower()}.png"
        )


    # ========================================================
    # 17. COMPUTATIONAL RESOURCE COMPARISON (1 PLOT)
    # ========================================================

    resource_metrics = [

        (
            "Training_Time_Seconds",
            "Mean Training Time (seconds)"
        ),

        (
            "Prediction_Time_Seconds",
            "Mean Prediction Time (seconds)"
        ),

        (
            "CPU_Time_Seconds",
            "Mean CPU Process Time (seconds)"
        ),

        (
            "Memory_Change_MB",
            "Mean Memory Change (MB)"
        )

    ]

    resource_means = (

        results_df.groupby("Hidden_Layers")[
            [item[0] for item in resource_metrics]
        ]

        .mean()

        .reset_index()

    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    for ax, (column, title) in zip(
        axes.ravel(),
        resource_metrics
    ):

        sns.lineplot(

            data=resource_means,

            x="Hidden_Layers",

            y=column,

            marker="o",

            ax=ax

        )

        ax.set_title(title)

        ax.set_xlabel(
            "Number of Hidden Layers"
        )

        ax.set_ylabel(title)

        ax.set_xticks(
            list(HIDDEN_LAYERS)
        )

        ax.grid(alpha=0.25)

    fig.suptitle(
        "Computational Resource Comparison"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    save_plot(
        fig,
        "resource_comparison_four_metrics.png"
    )


    # ========================================================
    # 18. SAVE EXPERIMENT SUMMARY
    # ========================================================

    summary_file = (
        OUTPUT_DIR / "classification_experiment_summary.txt"
    )

    with summary_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "DRY BEAN ANN CLASSIFICATION SUMMARY\n"
        )

        file.write(
            f"Dataset: {DATA_FILE}\n"
        )

        file.write(
            f"Samples: {len(X)}\n"
        )

        file.write(
            f"Features: {X.shape[1]}\n"
        )

        file.write(
            f"Classes: {class_names}\n"
        )

        file.write(
            "Split: 64% training / 16% validation / 20% testing\n"
        )

        file.write(
            f"Experiments completed: {len(results_df)}\n"
        )

        file.write(
            "\nBest validation configuration:\n"
        )

        file.write(
            best_result.to_string()
        )

        file.write(
            "\n\nHeld-out test metrics:\n"
        )

        file.write(
            str(test_metrics)
        )

        file.write(
            "\n\nNote: Scikit-learn uses the API token 'logistic' "
            "to implement the sigmoid activation. "
            "This experiment is labelled Sigmoid.\n"
        )


    # ========================================================
    # 19. FINAL OUTPUT SUMMARY
    # ========================================================

    plot_files = sorted(
        PLOTS_DIR.glob("*.png")
    )

    csv_files = sorted(
        OUTPUT_DIR.glob("*.csv")
    )

    print("\nHELD-OUT TEST RESULTS")
    print("-" * 60)

    for metric, value in test_metrics.items():

        print(
            f"{metric}: {value:.6f}"
        )

    print("\nEXPERIMENT COMPLETION")
    print("-" * 60)

    print(
        f"Experiments completed: {len(results_df)} / 63"
    )

    print(
        f"Plots created: {len(plot_files)} / 17"
    )

    for file in plot_files:

        print(
            " -",
            file.name
        )

    print(
        f"\nCSV files created: {len(csv_files)} / 6"
    )

    for file in csv_files:

        print(
            " -",
            file.name
        )

    print(
        f"\nAll results saved in: {OUTPUT_DIR}"
    )

    print(
        "\nANN Classification Experiment Completed!"
    )


# ============================================================
# 20. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
       
