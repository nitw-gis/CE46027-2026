# Dry Bean Classification Using Support Vector Machine (SVM) and Decision Tree

## 1. Project Overview

This project implements machine learning classification algorithms to identify different varieties of dry beans based on their physical and geometrical characteristics.

The project uses the Dry Bean Dataset, which contains measurements of dry beans extracted through computer vision techniques.

Two supervised machine learning algorithms were implemented and evaluated:

- Support Vector Machine (SVM)
- Decision Tree Classifier

Different SVM kernel functions were investigated to understand their influence on classification performance. The Decision Tree model was also developed and evaluated using a selected tree depth.

The main objective is to compare the classification performance of these algorithms and understand their suitability for multiclass classification problems.

## 2. Project Objectives

- Understand supervised machine learning classification.
- Explore the Dry Bean Dataset.
- Perform data preprocessing and feature standardization.
- Implement Support Vector Machine classification.
- Experiment with different SVM kernel functions.
- Implement Decision Tree classification.
- Evaluate classification performance using Accuracy and F1-score.
- Compare the performance of SVM and Decision Tree models.

## 3. Dataset Description

**Dataset:** Dry Bean Dataset

The dataset contains 13,611 observations and 16 numerical input features.

Each observation represents a dry bean sample described by its physical and geometrical measurements.

### Input Features

The dataset contains the following 16 features:

1. Area
2. Perimeter
3. Major Axis Length
4. Minor Axis Length
5. Aspect Ratio
6. Eccentricity
7. Convex Area
8. Equivalent Diameter
9. Extent
10. Solidity
11. Roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

### Target Variable

`Class`

The target variable represents seven different dry bean varieties:

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

The task is to classify each dry bean observation into its corresponding variety using the 16 input features.

## 4. Technologies and Libraries

The following technologies and Python libraries were used:

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- GitHub

## 5. Methodology

The project follows a supervised machine learning workflow.

### 5.1 Data Loading and Exploration

- Imported the Dry Bean Dataset from an Excel worksheet.
- Examined the dataset dimensions and data types.
- Inspected the input features and target variable.
- Analysed the distribution of the seven bean classes.

### 5.2 Data Preprocessing

- Separated independent variables (X) and the dependent variable (y).
- Prepared the 16 numerical input features.
- Converted categorical target labels into numerical representations.
- Divided the dataset into training and testing subsets.

### 5.3 Feature Standardization

StandardScaler was used to standardize the numerical input features.

Feature standardization is particularly important for SVM because the algorithm is sensitive to differences in feature scales.

The scaler was fitted using the training data and then applied to the testing data.

## 6. Support Vector Machine (SVM) Classification

Support Vector Machine is a supervised machine learning algorithm used for classification and regression problems.

In classification, SVM attempts to identify an optimal decision boundary that separates different classes.

The algorithm uses support vectors, which are the data points closest to the decision boundary, to determine the classification boundary.

### SVM Kernel Experimentation

Three SVM kernel functions were evaluated:

| Kernel | Description |
|---|---|
| Linear | Uses a linear decision boundary to separate classes. |
| Polynomial | Uses a polynomial transformation to model nonlinear relationships. |
| RBF | Radial Basis Function kernel, which can model complex nonlinear decision boundaries. |

### SVM Performance Results

| SVM Kernel | Accuracy | F1-score |
|---|---:|---:|
| Linear | 0.921004 | 0.921001 |
| Polynomial | 0.913990 | 0.915492 |
| RBF | 0.921004 | 0.921078 |

### SVM Analysis

- Linear and RBF kernels achieved the same accuracy of approximately 92.10%.
- The RBF kernel achieved the highest F1-score among the three SVM configurations.
- The Polynomial kernel produced comparatively lower accuracy and F1-score.
- The results demonstrate that kernel selection influences SVM classification performance.

## 7. Decision Tree Classification

Decision Tree is a supervised machine learning algorithm that classifies observations by learning a sequence of decision rules from the input features.

The model consists of:

- Root node
- Decision nodes
- Branches
- Leaf nodes

The algorithm selects features and splitting conditions to divide the dataset into increasingly homogeneous groups.

### Decision Tree Configuration

The Decision Tree classifier was implemented with a maximum tree depth of 5.

| Parameter | Configuration |
|---|---|
| Algorithm | Decision Tree Classifier |
| Maximum Depth | 5 |
| Task | Multiclass Classification |

### Decision Tree Performance

| Model | Configuration | Accuracy | F1-score |
|---|---|---:|---:|
| Decision Tree | Maximum Depth = 5 | 0.874862 | 0.875483 |

### Decision Tree Analysis

The Decision Tree achieved an accuracy of approximately 87.49% and an F1-score of approximately 87.55%.

The model demonstrated its ability to classify dry bean varieties using a sequence of feature-based decision rules.

## 8. Overall Model Performance Comparison

The following table compares all four evaluated model configurations.

| Model | Configuration | Accuracy | F1-score |
|---|---|---:|---:|
| SVM | Linear | 0.921004 | 0.921001 |
| SVM | Polynomial | 0.913990 | 0.915492 |
| SVM | RBF | 0.921004 | 0.921078 |
| Decision Tree | Depth = 5 | 0.874862 | 0.875483 |

### Comparative Analysis

**SVM – Linear Kernel**

The Linear SVM achieved 92.10% accuracy, demonstrating strong classification performance.

**SVM – Polynomial Kernel**

The Polynomial SVM achieved approximately 91.40% accuracy. It performed slightly lower than the Linear and RBF configurations.

**SVM – RBF Kernel**

The RBF SVM achieved 92.10% accuracy and the highest F1-score among the evaluated configurations.

**Decision Tree**

The Decision Tree achieved approximately 87.49% accuracy. Its performance was lower than the three tested SVM configurations.

### Overall Observation

Among the evaluated configurations, the SVM with the RBF kernel achieved the highest F1-score, while the Linear and RBF kernels shared the highest accuracy.

The results indicate that SVM performed better than the selected Decision Tree configuration on this dataset.

## 9. Evaluation Metrics

The following classification metrics were used to evaluate the models.

### Accuracy

Measures the proportion of correctly classified observations out of the total number of observations.

### Precision

Measures the proportion of correctly predicted positive observations among all observations predicted as positive for a particular class.

### Recall

Measures the proportion of actual positive observations correctly identified by the model.

### F1-score

Represents the harmonic mean of precision and recall.

It provides a balance between precision and recall and is useful when evaluating classification models.

## 10. Project Structure

```text
dry-bean-classification/
│
├── data/
│   └── Dry_Bean_Dataset.xlsx
│
├── notebook/
│   └── dry_bean_classification.ipynb
│
├── results/
│
├── figures/
│
├── README.md
│
└── requirements.txt
```

## 11. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/dry-bean-classification.git
```

### Step 2: Navigate to the Project Directory

```bash
cd dry-bean-classification
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Open Jupyter Notebook

```bash
jupyter notebook
```

Open the classification notebook inside the `notebook` directory and execute the cells sequentially.

## 12. Limitations and Future Scope

### Limitations

- The classification performance depends on the selected algorithm and its parameters.
- Only three SVM kernel configurations were evaluated.
- The Decision Tree was evaluated using a maximum depth of 5.
- Additional hyperparameter tuning may improve model performance.
- Model performance should be evaluated using consistent data splits and preprocessing procedures for fair comparison.

### Future Scope

- Experiment with additional SVM hyperparameters.
- Investigate different Decision Tree depths.
- Apply cross-validation for more reliable model evaluation.
- Explore additional classification algorithms.
- Compare model performance using confusion matrices and class-wise evaluation metrics.

## 13. Conclusion

This project demonstrates the application of supervised machine learning algorithms for multiclass classification of dry bean varieties.

The Support Vector Machine algorithm was evaluated using Linear, Polynomial, and RBF kernels, while a Decision Tree classifier was developed with a maximum depth of 5.

Among the evaluated configurations, the RBF SVM achieved the highest F1-score of 0.921078, while Linear and RBF SVM achieved the highest accuracy of 0.921004.

The Decision Tree achieved an accuracy of 0.874862.

The experimental results indicate that SVM provided better classification performance than the selected Decision Tree configuration.

Overall, this project provided practical experience in data preprocessing, feature standardization, supervised classification, SVM kernel experimentation, Decision Tree modelling, and model performance evaluation.

## Author

**Vyshnavi Perumal**

Artificial Intelligence and Machine Learning | Geoinformatics
