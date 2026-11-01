
"""
DRY BEAN CLASSIFICATION MODEL COMPARISON
Models:
1. Artificial Neural Network (ANN)
2. Support Vector Machine (SVM)
3. Decision Tree (DT)

ANN:
63 configurations:
- Hidden layers: 1 to 7
- Activation: ReLU, Sigmoid, Tanh
- Learning rates: 0.001, 0.005, 0.01
- 16 neurons per hidden layer

Dataset:
ANN_SVM_DT_Classification_Comparison/Dataset/Dry_Bean_Dataset.xlsx

Outputs:
ANN_SVM_DT_Classification_Comparison/output/
"""

from pathlib import Path
from itertools import product
import time
import warnings
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_FILE = PROJECT_DIR / "Dataset" / "Dry_Bean_Dataset.xlsx"

OUTPUT_DIR = PROJECT_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

RANDOM_STATE = 42

HIDDEN_LAYERS = range(1, 8)

ACTIVATIONS = ["ReLU", "Sigmoid", "Tanh"]

LEARNING_RATES = [0.001, 0.005, 0.01]

NEURONS = 16
MAX_ITER = 500


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_scores(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision_Weighted": precision_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall_Weighted": recall_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1_Weighted": f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),

        "Precision_Macro": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),

        "Recall_Macro": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),

        "F1_Macro": f1_score(
            y_true, y_pred,
            average="macro",
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
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

    X = df.drop(columns=[target_column]).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    if any(X[col].isna().all() for col in X.columns):
        raise ValueError(
            "At least one feature contains no numeric values."
        )

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        df[target_column].astype(str).str.strip()
    )

    class_names = list(label_encoder.classes_)

    print("\nDATASET INFORMATION")
    print("-" * 60)
    print("Dataset:", DATA_FILE.name)
    print("Samples:", len(df))
    print("Features:", X.shape[1])
    print("Classes:", class_names)
    print("Missing feature values:", int(X.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))

    print("\nClass distribution:")
    print(df[target_column].value_counts().sort_index())

    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST SPLIT
    # Same split procedure as the supplied ANN code.
    # --------------------------------------------------------

    all_indices = np.arange(len(X))

    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y[trainval_idx]
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print("\nDATA SPLIT")
    print("-" * 60)
    print("Training samples:", len(train_idx))
    print("Validation samples:", len(val_idx))
    print("Testing samples:", len(test_idx))

    # Save split indices for reproducibility.
    pd.DataFrame({
        "Original_Row_Index": train_idx,
        "Subset": "Training"
    }).to_csv(
        OUTPUT_DIR / "training_indices.csv",
        index=False
    )

    pd.DataFrame({
        "Original_Row_Index": val_idx,
        "Subset": "Validation"
    }).to_csv(
        OUTPUT_DIR / "validation_indices.csv",
        index=False
    )

    pd.DataFrame({
        "Original_Row_Index": test_idx,
        "Subset": "Testing"
    }).to_csv(
        OUTPUT_DIR / "testing_indices.csv",
        index=False
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    # ANN and SVM: median imputation + standardization.
    scaled_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_train_scaled = scaled_preprocessor.fit_transform(X_train)
    X_val_scaled = scaled_preprocessor.transform(X_val)
    X_test_scaled = scaled_preprocessor.transform(X_test)

    # Decision Tree: median imputation only.
    # Scaling is not required for tree-based models.
    tree_imputer = SimpleImputer(strategy="median")

    X_train_tree = tree_imputer.fit_transform(X_train)
    X_val_tree = tree_imputer.transform(X_val)
    X_test_tree = tree_imputer.transform(X_test)

    # ========================================================
    # 4. ANN: RUN 63 CONFIGURATIONS
    # ========================================================

    ann_results = []
    ann_models = {}

    configurations = list(product(
        HIDDEN_LAYERS,
        ACTIVATIONS,
        LEARNING_RATES
    ))

    print("\nRUNNING 63 ANN EXPERIMENTS")
    print("-" * 60)

    for number, (depth, activation, lr) in enumerate(
        configurations, start=1
    ):

        activation_api = {
            "ReLU": "relu",
            "Sigmoid": "logistic",
            "Tanh": "tanh"
        }[activation]

        model = MLPClassifier(
            hidden_layer_sizes=(NEURONS,) * depth,
            activation=activation_api,
            solver="adam",
            learning_rate_init=lr,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=25,
            tol=1e-4
        )

        start = time.perf_counter()

        model.fit(X_train_scaled, y_train)

        training_time = time.perf_counter() - start

        start = time.perf_counter()

        val_pred = model.predict(X_val_scaled)

        prediction_time = time.perf_counter() - start

        scores = calculate_scores(y_val, val_pred)

        name = (
            f"ANN_{depth}Layers_{activation}_LR_{lr}"
        )

        ann_results.append({
            "Model": name,
            "Hidden_Layers": depth,
            "Neurons_Per_Layer": NEURONS,
            "Activation": activation,
            "Learning_Rate": lr,
            "Iterations": model.n_iter_,
            **scores,
            "Training_Time_Seconds": training_time,
            "Validation_Prediction_Time_Seconds": prediction_time
        })

        ann_models[name] = model

        print(
            f"[{number:02d}/63] {name} | "
            f"Accuracy: {scores['Accuracy']:.4f} | "
            f"Weighted F1: {scores['F1_Weighted']:.4f}"
        )

        del val_pred
        gc.collect()

    ann_df = pd.DataFrame(ann_results)

    ann_df = ann_df.sort_values(
        by=[
            "F1_Weighted",
            "Accuracy",
            "Recall_Weighted",
            "Precision_Weighted"
        ],
        ascending=False
    ).reset_index(drop=True)

    ann_df.insert(
        0,
        "Validation_Rank",
        np.arange(1, len(ann_df) + 1)
    )

    ann_df.to_csv(
        OUTPUT_DIR / "ann_63_experiments.csv",
        index=False
    )

    best_ann_name = ann_df.iloc[0]["Model"]
    best_ann = ann_models[best_ann_name]

    print("\nBEST ANN CONFIGURATION")
    print("-" * 60)
    print(ann_df.iloc[0].to_string())

    # ========================================================
    # 5. SVM CLASSIFIER
    # ========================================================

    print("\nTRAINING SVM")
    print("-" * 60)

    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    start = time.perf_counter()

    svm.fit(X_train_scaled, y_train)

    svm_training_time = time.perf_counter() - start

    # ========================================================
    # 6. DECISION TREE CLASSIFIER
    # ========================================================

    print("\nTRAINING DECISION TREE")
    print("-" * 60)

    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        random_state=RANDOM_STATE
    )

    start = time.perf_counter()

    decision_tree.fit(X_train_tree, y_train)

    dt_training_time = time.perf_counter() - start

    # ========================================================
    # 7. VALIDATION RESULTS FOR SVM AND DT
    # ========================================================

    # Validation results are reported for reference.
    # No test data is used for model selection.

    svm_val_pred = svm.predict(X_val_scaled)
    dt_val_pred = decision_tree.predict(X_val_tree)

    svm_val_scores = calculate_scores(y_val, svm_val_pred)
    dt_val_scores = calculate_scores(y_val, dt_val_pred)

    # ========================================================
    # 8. FINAL HELD-OUT TEST EVALUATION
    # ========================================================

    print("\nEVALUATING ALL MODELS ON TEST DATA")
    print("-" * 60)

    model_predictions = {}
    model_objects = {
        "ANN": best_ann,
        "SVM": svm,
        "Decision Tree": decision_tree
    }

    test_inputs = {
        "ANN": X_test_scaled,
        "SVM": X_test_scaled,
        "Decision Tree": X_test_tree
    }

    training_times = {
        "ANN": float(
            ann_df.iloc[0]["Training_Time_Seconds"]
        ),
        "SVM": svm_training_time,
        "Decision Tree": dt_training_time
    }

    comparison_results = []

    for model_name, model in model_objects.items():

        start = time.perf_counter()

        predictions = model.predict(
            test_inputs[model_name]
        )

        prediction_time = time.perf_counter() - start

        model_predictions[model_name] = predictions

        scores = calculate_scores(y_test, predictions)

        comparison_results.append({
            "Model": model_name,
            **scores,
            "Training_Time_Seconds": training_times[model_name],
            "Test_Prediction_Time_Seconds": prediction_time
        })

        print(f"\n{model_name}")
        for metric, value in scores.items():
            print(f"{metric}: {value:.6f}")

    comparison_df = pd.DataFrame(comparison_results)

    comparison_df.to_csv(
        OUTPUT_DIR / "classification_model_comparison.csv",
        index=False
    )

    # ========================================================
    # 9. MODEL PERFORMANCE RANKING
    # ========================================================

    ranking = comparison_df[["Model"]].copy()

    ranking["Accuracy_Rank"] = (
        comparison_df["Accuracy"].rank(
            ascending=False,
            method="min"
        )
    )

    ranking["Precision_Rank"] = (
        comparison_df["Precision_Weighted"].rank(
            ascending=False,
            method="min"
        )
    )

    ranking["Recall_Rank"] = (
        comparison_df["Recall_Weighted"].rank(
            ascending=False,
            method="min"
        )
    )

    ranking["F1_Rank"] = (
        comparison_df["F1_Weighted"].rank(
            ascending=False,
            method="min"
        )
    )

    rank_columns = [
        "Accuracy_Rank",
        "Precision_Rank",
        "Recall_Rank",
        "F1_Rank"
    ]

    ranking["Average_Rank"] = ranking[rank_columns].mean(axis=1)

    ranking["Overall_Rank"] = ranking[
        "Average_Rank"
    ].rank(
        ascending=True,
        method="min"
    ).astype(int)

    ranking = ranking.sort_values(
        "Overall_Rank"
    )

    ranking.to_csv(
        OUTPUT_DIR / "classification_performance_ranking.csv",
        index=False
    )

    # ========================================================
    # 10. PER-CLASS METRICS
    # ========================================================

    per_class_rows = []

    for model_name, predictions in model_predictions.items():

        report = classification_report(
            y_test,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        for class_name in class_names:

            per_class_rows.append({
                "Model": model_name,
                "Class": class_name,
                "Precision": report[class_name]["precision"],
                "Recall": report[class_name]["recall"],
                "F1_Score": report[class_name]["f1-score"],
                "Support": int(report[class_name]["support"])
            })

    pd.DataFrame(per_class_rows).to_csv(
        OUTPUT_DIR / "classification_per_class_metrics.csv",
        index=False
    )

    # ========================================================
    # 11. SAVE TEST PREDICTIONS
    # ========================================================

    prediction_data = {
        "Original_Row_Index": X_test.index,
        "Actual_Class": label_encoder.inverse_transform(y_test)
    }

    for model_name, predictions in model_predictions.items():

        safe_name = model_name.replace(" ", "_")

        prediction_data[f"{safe_name}_Prediction"] = (
            label_encoder.inverse_transform(predictions)
        )

        prediction_data[f"{safe_name}_Correct"] = (
            predictions == y_test
        )

    pd.DataFrame(prediction_data).to_csv(
        OUTPUT_DIR / "classification_test_predictions.csv",
        index=False
    )

    # ========================================================
    # 12. CONFUSION MATRICES
    # ========================================================

    for model_name, predictions in model_predictions.items():

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=np.arange(len(class_names))
        )

        fig, ax = plt.subplots(figsize=(10, 8))

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
            f"{model_name} - Test Confusion Matrix"
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        plt.setp(
            ax.get_xticklabels(),
            rotation=35,
            ha="right"
        )

        filename = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        save_plot(
            fig,
            f"confusion_matrix_{filename}.png"
        )

    # ========================================================
    # 13. METRIC COMPARISON PLOTS
    # ========================================================

    metric_plots = [
        ("Accuracy", "Accuracy"),
        ("Precision_Weighted", "Weighted Precision"),
        ("Recall_Weighted", "Weighted Recall"),
        ("F1_Weighted", "Weighted F1-score")
    ]

    for column, title in metric_plots:

        fig, ax = plt.subplots(figsize=(9, 6))

        sns.barplot(
            data=comparison_df,
            x="Model",
            y=column,
            ax=ax
        )

        ax.set_title(f"{title} Comparison")
        ax.set_xlabel("Classification Model")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)

        save_plot(
            fig,
            f"comparison_{column.lower()}.png"
        )

    # ========================================================
    # 14. COMPUTATIONAL TIME COMPARISON
    # ========================================================

    time_df = comparison_df.melt(
        id_vars="Model",
        value_vars=[
            "Training_Time_Seconds",
            "Test_Prediction_Time_Seconds"
        ],
        var_name="Time_Type",
        value_name="Time_Seconds"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=time_df,
        x="Model",
        y="Time_Seconds",
        hue="Time_Type",
        ax=ax
    )

    ax.set_title("Training and Prediction Time Comparison")
    ax.set_xlabel("Classification Model")
    ax.set_ylabel("Time (seconds)")
    ax.grid(axis="y", alpha=0.25)

    save_plot(
        fig,
        "comparison_computational_time.png"
    )

    # ========================================================
    # 15. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("CLASSIFICATION MODEL COMPARISON")
    print("=" * 65)

    print(
        comparison_df.to_string(index=False)
    )

    print("\nOVERALL PERFORMANCE RANKING")
    print("-" * 65)

    print(
        ranking.to_string(index=False)
    )

    print("\nBEST MODEL BY ACCURACY")

    best_accuracy = comparison_df.loc[
        comparison_df["Accuracy"].idxmax()
    ]

    print(
        best_accuracy[
            ["Model", "Accuracy"]
        ].to_string()
    )

    print("\nBEST MODEL BY WEIGHTED F1-SCORE")

    best_f1 = comparison_df.loc[
        comparison_df["F1_Weighted"].idxmax()
    ]

    print(
        best_f1[
            ["Model", "F1_Weighted"]
        ].to_string()
    )

    print("\nOUTPUT SUMMARY")
    print("-" * 65)

    csv_files = sorted(OUTPUT_DIR.glob("*.csv"))
    plot_files = sorted(PLOTS_DIR.glob("*.png"))

    print("CSV files generated:", len(csv_files))

    for file in csv_files:
        print("-", file.name)

    print("\nPlots generated:", len(plot_files))

    for file in plot_files:
        print("-", file.name)

    print("\nAll outputs saved in:")
    print(OUTPUT_DIR)

    print("\nCLASSIFICATION MODEL COMPARISON COMPLETED!")


# ============================================================
# 16. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
