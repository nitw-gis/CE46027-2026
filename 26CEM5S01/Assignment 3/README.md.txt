# Assignment 3 – Classification

## 1. Objective

The objective of this assignment is to perform multiclass classification using Support Vector Machine (SVM) and Decision Tree classifiers on the Dry Bean Dataset. Three SVM kernels, namely Linear, Polynomial and RBF, are evaluated. Decision Tree performance is investigated by varying the `max_depth` and `min_samples_split` parameters. The models are evaluated using classification metrics and computational performance measures.

## 2. Dataset

The Dry Bean Dataset was obtained from Kaggle.

The dataset contains physical and shape-related characteristics of dry bean samples. It consists of 16 predictor variables and one target variable representing the bean class.

The dataset contains 13,611 samples and 17 columns in total.

## 3. Software and Libraries

- Python 3.11.5
- Python IDLE
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- psutil
- openpyxl

## 4. Methodology

### 4.1 Data Loading

The Dry Bean dataset was loaded from an Excel file using the pandas library.

### 4.2 Data Inspection and Preprocessing

The dataset was inspected for its dimensions, column names, missing values, and class distribution. The target variable `Class` was separated from the predictor variables.

The categorical target classes were converted into numerical labels using `LabelEncoder`.

### 4.3 Train-Test Split

The dataset was divided into training and testing datasets using an 80:20 ratio. Stratified sampling was applied to maintain the class distribution in both subsets.

A fixed `random_state=42` was used to ensure reproducibility.

### 4.4 Feature Scaling

`StandardScaler` was applied to the feature variables used for SVM classification. The scaler was fitted using the training data and then applied to both training and testing data.

Decision Tree models were trained using the original unscaled feature values.

### 4.5 t-SNE Visualization

t-SNE was applied to the standardized training feature data to obtain a two-dimensional representation of the feature space.

The resulting two-dimensional representation was visualized using a scatter plot, with different colors representing the dry bean classes.

### 4.6 SVM Classification

Three Support Vector Machine models were implemented:

1. SVM with Linear Kernel
2. SVM with Polynomial Kernel
3. SVM with RBF Kernel

For the Polynomial kernel, a polynomial degree of 3 was used.

### 4.7 Decision Tree Classification

Decision Tree classifiers were evaluated by varying two structural parameters.

For `max_depth`, the following values were tested:

- 3
- 5
- 10
- 15
- None

For `min_samples_split`, the following values were tested:

- 2
- 5
- 10
- 20
- 50

A fixed `random_state=42` was used for the Decision Tree models.

### 4.8 Performance Evaluation

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Execution time
- Memory change

Weighted averaging was used for Precision, Recall, and F1-score.

## 5. Experimental Results

The following results were obtained from the actual execution of the classification models.

| Model | Accuracy | Precision | Recall | F1-Score | Execution Time (s) | Memory Change (MB) |
|---|---:|---:|---:|---:|---:|---:|
| SVM - Linear Kernel | 0.922145 | 0.922227 | 0.922145 | 0.922107 | 0.482839 | 2.667969 |
| SVM - Polynomial Kernel | 0.914433 | 0.921983 | 0.914433 | 0.915887 | 0.578103 | 1.097656 |
| SVM - RBF Kernel | 0.922145 | 0.922426 | 0.922145 | 0.922177 | 0.855110 | -0.730469 |
| Decision Tree - max_depth=3 | 0.773044 | 0.716308 | 0.773044 | 0.732263 | 0.054065 | 1.300781 |
| Decision Tree - max_depth=5 | 0.881014 | 0.885351 | 0.881014 | 0.880838 | 0.093454 | 0.000000 |
| Decision Tree - max_depth=10 | 0.905619 | 0.905952 | 0.905619 | 0.905694 | 0.156396 | 0.015625 |
| Decision Tree - max_depth=15 | 0.896438 | 0.896748 | 0.896438 | 0.896507 | 0.168431 | 0.000000 |
| Decision Tree - max_depth=None | 0.891664 | 0.891247 | 0.891664 | 0.891322 | 0.215183 | 0.007812 |
| Decision Tree - min_samples_split=2 | 0.891664 | 0.891247 | 0.891664 | 0.891322 | 0.205276 | 0.000000 |
| Decision Tree - min_samples_split=5 | 0.894234 | 0.893843 | 0.894234 | 0.893888 | 0.186867 | 0.000000 |
| Decision Tree - min_samples_split=10 | 0.898274 | 0.897963 | 0.898274 | 0.897954 | 0.216150 | 0.000000 |
| Decision Tree - min_samples_split=20 | 0.901212 | 0.900958 | 0.901212 | 0.900810 | 0.164406 | 0.000000 |
| Decision Tree - min_samples_split=50 | 0.904150 | 0.903744 | 0.904150 | 0.903782 | 0.147424 | 0.000000 |

## 6. Results Discussion

### 6.1 SVM Results

The three SVM kernels produced accuracy values between 0.914433 and 0.922145.

The Linear and RBF kernels both produced an accuracy of 0.922145. Their F1-scores were 0.922107 and 0.922177, respectively.

The Polynomial kernel produced an accuracy of 0.914433 and an F1-score of 0.915887.

The recorded execution times were 0.482839 seconds for the Linear kernel, 0.578103 seconds for the Polynomial kernel, and 0.855110 seconds for the RBF kernel.

### 6.2 Decision Tree Results – max_depth

The Decision Tree accuracy increased from 0.773044 at `max_depth=3` to 0.905619 at `max_depth=10`.

For `max_depth=15`, the accuracy was 0.896438, while `max_depth=None` produced an accuracy of 0.891664.

The results show that changing the maximum tree depth affected both classification performance and execution time in the experiments.

### 6.3 Decision Tree Results – min_samples_split

When `min_samples_split` was varied from 2 to 50, the observed accuracy values were:

- `min_samples_split=2`: 0.891664
- `min_samples_split=5`: 0.894234
- `min_samples_split=10`: 0.898274
- `min_samples_split=20`: 0.901212
- `min_samples_split=50`: 0.904150

The corresponding F1-scores ranged from 0.891322 to 0.903782.

### 6.4 Computational Performance

The Decision Tree experiments generally required less execution time than the SVM experiments in this experiment.

The recorded execution times for the Decision Tree models ranged from 0.054065 to 0.216150 seconds, whereas the SVM execution times ranged from 0.482839 to 0.855110 seconds.

Memory change was measured using the process memory before and after model execution. The values represent the recorded change in process memory during each experiment and may vary depending on the system state during execution.

## 7. Visualizations and Output Files

The following output files were generated during the experiment:

- `class_distribution.png` – Distribution of the dry bean classes
- `tsne_plot.png` – Two-dimensional t-SNE visualization
- `confusion_matrix_svm_linear.png` – Confusion matrix for the Linear SVM
- `confusion_matrix_svm_poly.png` – Confusion matrix for the Polynomial SVM
- `confusion_matrix_svm_rbf.png` – Confusion matrix for the RBF SVM
- `accuracy_comparison.png` – Accuracy comparison of all classification models
- `f1_comparison.png` – F1-score comparison of all classification models
- `execution_time_comparison.png` – Execution time comparison of all classification models
- `model_comparison.csv` – Numerical comparison of all evaluated models

## 8. Conclusion

This assignment demonstrated the application of SVM and Decision Tree classifiers to the multiclass Dry Bean classification problem.

Three SVM kernels and multiple Decision Tree parameter settings were evaluated using classification and computational performance measures. The experimental results showed differences in accuracy, precision, recall, F1-score, execution time, and memory change among the tested configurations.

The SVM models achieved accuracy values above 0.91, while the Decision Tree models showed accuracy values ranging from 0.773044 to 0.905619 depending on the selected parameters.

The experiment also demonstrated how SVM kernel selection and Decision Tree structural parameters can affect classification performance and computational requirements.

## 9. Project Structure

```text
Assignment3/
│
├── dataset/
│   └── Dry_Bean_Dataset.xlsx
│
├── code/
│   └── assignment3_classification.py
│
├── outputs/
│   ├── class_distribution.png
│   ├── tsne_plot.png
│   ├── confusion_matrix_svm_linear.png
│   ├── confusion_matrix_svm_poly.png
│   ├── confusion_matrix_svm_rbf.png
│   ├── accuracy_comparison.png
│   ├── f1_comparison.png
│   ├── execution_time_comparison.png
│   └── model_comparison.csv
│
└── README.md