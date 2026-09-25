Dry Bean Classification Using Machine Learning

Project Overview

This project implements a machine learning classification workflow using the UCI Dry Bean Dataset to classify dry bean varieties based on morphological and shape-related characteristics.

Two supervised machine learning approaches are investigated:

Support Vector Machine (SVM)

Decision Tree Classifier

For SVM, three kernels are evaluated:

Linear Kernel

Polynomial Kernel

RBF Kernel

For the Decision Tree, multiple combinations of max_depth and min_samples_split are tested.

The models are evaluated using Accuracy, Precision, Recall, F1 Score, Training Time, Prediction Time, and Confusion Matrices.

Objectives

Load and inspect the Dry Bean dataset.

Check for missing values and duplicate records.

Analyze the distribution of bean classes.

Separate features and target variables.

Encode categorical class labels.

Split the dataset into training and testing sets.

Standardize the features for SVM modeling.

Visualize the dataset using t-SNE.

Implement Linear, Polynomial, and RBF SVM classifiers.

Compare the performance of different SVM kernels.

Implement a Decision Tree classifier.

Test multiple Decision Tree parameter combinations.

Compare SVM and Decision Tree performance.

Generate visualizations and confusion matrices.

Save model evaluation results as CSV files.

Dataset

The project uses the UCI Dry Bean Dataset, which contains measurements describing different varieties of dry beans.

Dataset Information

Original observations: 13,611

Columns: 17

Numerical features: 16

Target variable: Class

Number of classes: 7

After removing duplicate records:

Final observations: 13,543

Features: 16

Target classes: 7

Bean Classes

BARBUNYA

BOMBAY

CALI

DERMASON

HOROZ

SEKER

SIRA

Class Distribution

Class

Samples

DERMASON

3546

SIRA

2636

SEKER

2027

HOROZ

1928

CALI

1630

BARBUNYA

1322

BOMBAY

522

Data Preprocessing

Missing Values

The dataset was checked for missing values using:

df.isnull().sum()

No missing values were found.

Duplicate Records

A total of 68 duplicate records were identified and removed:

df = df.drop_duplicates().reset_index(drop=True)

The resulting dataset contains 13,543 observations.

Technologies Used

Python

Jupyter Notebook

Pandas

NumPy

Matplotlib

Scikit-learn

OpenPyXL

GitHub

Project Structure

Assignment3_DryBean/
│
├── Assignment3_DryBean.ipynb
├── data/
│   └── Dry_Bean_Dataset.xlsx
├── figures/
│   ├── class_distribution.png
│   ├── tsne_dry_bean.png
│   ├── confusion_matrix_linear_svm.png
│   ├── confusion_matrix_polynomial_svm.png
│   ├── confusion_matrix_rbf_svm.png
│   ├── svm_kernel_comparison.png
│   ├── decision_tree_depth_comparison.png
│   ├── decision_tree_f1_comparison.png
│   ├── confusion_matrix_decision_tree.png
│   └── final_model_comparison.png
├── results/
│   ├── SVM_Comparison.csv
│   ├── Decision_Tree_Comparison.csv
│   └── Final_Model_Comparison.csv
├── README.md
├── requirements.txt
└── .gitignore

Machine Learning Workflow

1. Data Loading

df = pd.read_excel("../data/Dry_Bean_Dataset.xlsx")

2. Data Inspection

df.head()
df.shape
df.columns
df.dtypes
df.isnull().sum()
df.duplicated().sum()

3. Feature and Target Separation

X = df.drop("Class", axis=1)
y = df["Class"]

The dataset contains 16 input features and 1 target variable.

4. Label Encoding

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

5. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

Training samples: 10,834

Testing samples: 2,709

6. Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

t-SNE Visualization

t-SNE was used to reduce the high-dimensional feature space to two dimensions for visualization. A stratified sample of 5,000 training observations was used.

X_tsne, _, y_tsne, _ = train_test_split(
    X_train_scaled,
    y_train,
    train_size=5000,
    random_state=42,
    stratify=y_train
)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=1000,
    init="pca"
)

X_tsne_2d = tsne.fit_transform(X_tsne)

Output:

figures/tsne_dry_bean.png

Support Vector Machine

Three SVM kernels were evaluated.

Linear Kernel

SVC(kernel="linear", random_state=42)

Polynomial Kernel

SVC(kernel="poly", degree=3, random_state=42)

RBF Kernel

SVC(kernel="rbf", random_state=42)

SVM Results

Model

Accuracy

Precision

Recall

F1 Score

SVM - Linear

0.921004

0.921098

0.921004

0.921001

SVM - Polynomial

0.913990

0.921894

0.913990

0.915492

SVM - RBF

0.921004

0.921411

0.921004

0.921078

Complete results:

results/SVM_Comparison.csv

SVM Confusion Matrices

figures/confusion_matrix_linear_svm.png
figures/confusion_matrix_polynomial_svm.png
figures/confusion_matrix_rbf_svm.png

SVM comparison chart:

figures/svm_kernel_comparison.png

Decision Tree

The Decision Tree was evaluated using:

Maximum Depth

5
10
15
20

Minimum Samples Split

2
10
20

A total of 12 configurations were tested.

Decision Tree Results

Complete comparison:

results/Decision_Tree_Comparison.csv

Generated visualizations:

figures/decision_tree_depth_comparison.png
figures/decision_tree_f1_comparison.png

Selected Decision Tree Configuration

For detailed evaluation, the following tested configuration was selected:

max_depth = 15
min_samples_split = 20

Performance

Metric

Value

Accuracy

0.908822

Precision

0.908626

Recall

0.908822

F1 Score

0.908552

Decision Tree confusion matrix:

figures/confusion_matrix_decision_tree.png

Final Model Comparison

Model

Accuracy

Precision

Recall

F1 Score

SVM - Linear

0.921004

0.921098

0.921004

0.921001

SVM - Polynomial

0.913990

0.921894

0.913990

0.915492

SVM - RBF

0.921004

0.921411

0.921004

0.921078

Decision Tree

0.908822

0.908626

0.908822

0.908552

Final comparison file:

results/Final_Model_Comparison.csv

Final comparison visualization:

figures/final_model_comparison.png

Computational Performance

Model

Training Time (s)

Prediction Time (s)

SVM - Linear

1.384946

0.481592

SVM - Polynomial

2.169969

0.786913

SVM - RBF

2.004742

2.029755

Decision Tree

0.714068

0.006538

These measurements depend on the computational environment used for the experiment.

Evaluation Metrics

Accuracy

Accuracy measures the proportion of correctly classified observations.

Accuracy = Correct Predictions / Total Predictions

Precision

Precision measures how many observations predicted as a particular class actually belong to that class.

Recall

Recall measures how many observations belonging to a class were correctly identified.

F1 Score

F1 Score is the harmonic mean of precision and recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Weighted averages were used for Precision, Recall, and F1 Score.

Output Files

SVM Results

results/SVM_Comparison.csv

Decision Tree Results

results/Decision_Tree_Comparison.csv

Final Model Comparison

results/Final_Model_Comparison.csv

Generated Figures

figures/class_distribution.png
figures/tsne_dry_bean.png
figures/confusion_matrix_linear_svm.png
figures/confusion_matrix_polynomial_svm.png
figures/confusion_matrix_rbf_svm.png
figures/svm_kernel_comparison.png
figures/decision_tree_depth_comparison.png
figures/decision_tree_f1_comparison.png
figures/confusion_matrix_decision_tree.png
figures/final_model_comparison.png

Installation

Install the required Python packages using:

pip install -r requirements.txt

Requirements

pandas
numpy
matplotlib
scikit-learn
openpyxl
jupyter

How to Run

Open the Assignment3_DryBean project folder.

Start Jupyter Notebook:

jupyter notebook

Open Assignment3_DryBean.ipynb.

Run all cells from beginning to end.

The notebook performs data loading, preprocessing, t-SNE visualization, SVM training, Decision Tree parameter testing, evaluation, confusion matrix generation, model comparison, and result generation.

Reproducibility

The experiments use:

random_state = 42

This helps make the data split and results reproducible under the same software and computational environment.

Conclusion

This project demonstrates a complete supervised machine learning workflow for classifying dry bean varieties using morphological features.

The workflow includes:

Data preprocessing

Exploratory data analysis

Feature scaling

t-SNE visualization

Support Vector Machine classification

Decision Tree classification

Hyperparameter comparison

Performance evaluation

Confusion matrix analysis

Visualization

CSV result generation

The tested SVM configurations achieved accuracies between approximately 91.4% and 92.1%, while the selected Decision Tree configuration achieved an accuracy of approximately 90.9%.

The project provides a reproducible implementation of machine learning classification techniques using Python and Scikit-learn.

Author

Vyshnavi Perumal

GitHub Topics

machine-learning
python
scikit-learn
classification
dry-bean-dataset
svm
decision-tree
data-science
jupyter-notebook
t-sne
supervised-learning
