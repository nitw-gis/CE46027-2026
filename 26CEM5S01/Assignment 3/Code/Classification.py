import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import time
import psutil
import os


file_path = r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 3\Dataset\Dry_Bean_Dataset\Dry_Bean_Dataset.xlsx"

output_path = r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 3\outputs"

os.makedirs(
    output_path,
    exist_ok=True
)


df = pd.read_excel(file_path)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())


plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Class",
    order=df["Class"].value_counts().index
)

plt.title("Distribution of Dry Bean Classes")
plt.xlabel("Bean Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_path,
        "class_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nEncoded classes:")
print(label_encoder.classes_)

print("\nClass encoding:")

for number, class_name in enumerate(
    label_encoder.classes_
):
    print(
        number,
        "=",
        class_name
    )


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining feature shape:")
print(X_train.shape)

print("\nTesting feature shape:")
print(X_test.shape)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(
    X_train_scaled
)

print("t-SNE completed.")


plt.figure(
    figsize=(12, 8)
)

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y_train,
    cmap="tab10",
    s=10,
    alpha=0.7
)

plt.title(
    "2D t-SNE Visualization of Dry Bean Classes"
)

plt.xlabel(
    "t-SNE Component 1"
)

plt.ylabel(
    "t-SNE Component 2"
)

handles, _ = scatter.legend_elements()

plt.legend(
    handles,
    label_encoder.classes_,
    title="Bean Class",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_path,
        "tsne_plot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


def evaluate_model(
    model,
    X_train_data,
    X_test_data,
    y_train_data,
    y_test_data,
    model_name
):

    process = psutil.Process()

    memory_before = (
        process.memory_info().rss
        / (1024 ** 2)
    )

    start_time = time.perf_counter()

    model.fit(
        X_train_data,
        y_train_data
    )

    y_pred = model.predict(
        X_test_data
    )

    end_time = time.perf_counter()

    memory_after = (
        process.memory_info().rss
        / (1024 ** 2)
    )

    execution_time = (
        end_time - start_time
    )

    memory_used = (
        memory_after - memory_before
    )

    accuracy = accuracy_score(
        y_test_data,
        y_pred
    )

    precision = precision_score(
        y_test_data,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test_data,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test_data,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test_data,
        y_pred
    )

    print("\n" + model_name)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Execution Time:", execution_time, "seconds")
    print("Memory Change:", memory_used, "MB")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test_data,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Execution Time (s)": execution_time,
        "Memory Change (MB)": memory_used,
        "Confusion Matrix": cm
    }


svm_linear = SVC(
    kernel="linear",
    random_state=42
)

result_linear = evaluate_model(
    svm_linear,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    "SVM - Linear Kernel"
)


svm_poly = SVC(
    kernel="poly",
    degree=3,
    random_state=42
)

result_poly = evaluate_model(
    svm_poly,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    "SVM - Polynomial Kernel"
)


svm_rbf = SVC(
    kernel="rbf",
    random_state=42
)

result_rbf = evaluate_model(
    svm_rbf,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    "SVM - RBF Kernel"
)


def plot_confusion_matrix(
    cm,
    title,
    filename
):

    plt.figure(
        figsize=(9, 7)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


plot_confusion_matrix(
    result_linear["Confusion Matrix"],
    "Confusion Matrix - SVM Linear Kernel",
    os.path.join(
        output_path,
        "confusion_matrix_svm_linear.png"
    )
)


plot_confusion_matrix(
    result_poly["Confusion Matrix"],
    "Confusion Matrix - SVM Polynomial Kernel",
    os.path.join(
        output_path,
        "confusion_matrix_svm_poly.png"
    )
)


plot_confusion_matrix(
    result_rbf["Confusion Matrix"],
    "Confusion Matrix - SVM RBF Kernel",
    os.path.join(
        output_path,
        "confusion_matrix_svm_rbf.png"
    )
)

depth_values = [3, 5, 10, 15, None]

decision_tree_results = []

for depth in depth_values:

    dt = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    result = evaluate_model(
        dt,
        X_train,
        X_test,
        y_train,
        y_test,
        f"Decision Tree - max_depth={depth}"
    )

    decision_tree_results.append(result)

    split_values = [2, 5, 10, 20, 50]

for split in split_values:

    dt = DecisionTreeClassifier(
        min_samples_split=split,
        random_state=42
    )

    result = evaluate_model(
        dt,
        X_train,
        X_test,
        y_train,
        y_test,
        f"Decision Tree - min_samples_split={split}"
    )

    decision_tree_results.append(result)

    all_results = [
    result_linear,
    result_poly,
    result_rbf
] + decision_tree_results

    results_df = pd.DataFrame(all_results)

    print("\n\nFINAL MODEL COMPARISON")

print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Execution Time (s)",
            "Memory Change (MB)"
        ]
    ].to_string(index=False)
)

results_df.to_csv(
    os.path.join(
        output_path,
        "model_comparison.csv"
    ),
    index=False
)

print("\nModel comparison saved successfully.")

plt.figure(figsize=(14, 7))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Classification Models")

plt.xticks(rotation=75)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_path,
        "accuracy_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(14, 7))

plt.bar(
    results_df["Model"],
    results_df["F1-Score"]
)

plt.xlabel("Model")
plt.ylabel("F1-Score")
plt.title("F1-Score Comparison of Classification Models")

plt.xticks(rotation=75)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_path,
        "f1_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(14, 7))

plt.bar(
    results_df["Model"],
    results_df["Execution Time (s)"]
)

plt.xlabel("Model")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time Comparison of Classification Models")

plt.xticks(rotation=75)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_path,
        "execution_time_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()
