# Artificial Neural Network (ANN) Classification

## 1. Project overview

This project implements an Artificial Neural Network (ANN) for a
supervised classification task. The model is trained and evaluated
using a labelled tabular dataset.

The experiments compare three activation functions and three learning
rates while keeping the seven-hidden-layer network architecture and
the train/test split fixed.

## 2. Dataset

- **Dataset file:** The dataset selected when running the Python script.
- **Target column:** `diagnosis` (as configured in the ANN script).
- **Input features:** Numerical predictor columns; ID-like columns are excluded.
- **Task:** Classification.

Before submission, confirm that this dataset description and target
column match the dataset you actually used.

## 3. Software and libraries

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenPyXL
- psutil
- Spyder IDE (development environment)

Install the required libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl psutil
```

In the Spyder IPython Console, you can use `%pip install` followed by
the same package names.

## 4. Methodology

### 4.1 Data preprocessing

1. Load the Excel or CSV dataset.
2. Remove rows with missing target labels.
3. Exclude ID-like columns from the predictor variables.
4. Convert predictor columns to numerical values.
5. Fill remaining missing numerical values using column medians.
6. Encode the target classes into numerical labels.
7. Split the data into training and testing sets using an 80:20 ratio.
8. Standardize the features. The scaler is fitted only on the training
   data and then applied to the test data.

### 4.2 t-SNE visualization

A two-dimensional t-SNE representation is generated from the
standardized feature space and coloured by target class. The plot is
used to visually inspect the distribution and separation of classes
before ANN training. t-SNE is used for visualization only; its two
coordinates are not used as ANN input features.

### 4.3 ANN architecture

The ANN uses seven hidden layers:

- Hidden layer 1: 128 neurons
- Hidden layer 2: 64 neurons
- Hidden layer 3: 64 neurons
- Hidden layer 4: 32 neurons
- Hidden layer 5: 32 neurons
- Hidden layer 6: 16 neurons
- Hidden layer 7: 8 neurons

The output layer predicts the target classes. The implementation uses
the Adam optimization algorithm.

### 4.4 Experimental design

Nine experiments are performed using all combinations of:

**Activation functions**
- ReLU
- Tanh
- Logistic (sigmoid)

**Learning rates**
- 0.0001
- 0.001
- 0.01

The same training/testing split and network architecture are used
across experiments to support a fair comparison.

## 5. Evaluation metrics

The models are evaluated using:

- **Accuracy:** Proportion of all test predictions that are correct.
- **Precision (weighted):** Weighted average of class-wise precision.
- **Recall (weighted):** Weighted average of class-wise recall.
- **F1-score (weighted):** Weighted average of class-wise F1-scores.
- **Confusion matrix:** Counts of correct and incorrect predictions
  for each class.

Weighted metrics account for the number of test samples in each class.

## 6. Computational resource utilization

The script records execution time, CPU time, sampled CPU utilization,
and sampled process RAM usage for each experiment.

- **Average CPU utilization (%):** 174.43
- **Maximum sampled CPU (%):** 249.30
- **Peak sampled RAM (MB):** 424.29
- **CPU time (seconds):** 0.70

These values depend on the computer, background processes, software
environment, and dataset. They should be interpreted as measurements
from the machine on which the script was run.

## 7. Experimental results

The following table is read directly from `ANN_Results.csv` and sorted
by weighted F1-score (highest first).

| Experiment | Activation | Learning rate | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) | Execution time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ANN_tanh_LR_0.001 | tanh | 0.001 | 0.9825 | 0.9829 | 0.9825 | 0.9824 | 0.403 |
| ANN_tanh_LR_0.01 | tanh | 0.01 | 0.9737 | 0.9747 | 0.9737 | 0.9735 | 0.503 |
| ANN_relu_LR_0.01 | relu | 0.01 | 0.9737 | 0.9747 | 0.9737 | 0.9735 | 0.303 |
| ANN_logistic_LR_0.01 | logistic | 0.01 | 0.9649 | 0.9668 | 0.9649 | 0.9645 | 0.604 |
| ANN_relu_LR_0.001 | relu | 0.001 | 0.9561 | 0.9590 | 0.9561 | 0.9555 | 0.403 |
| ANN_tanh_LR_0.0001 | tanh | 0.0001 | 0.9474 | 0.9474 | 0.9474 | 0.9471 | 0.503 |
| ANN_relu_LR_0.0001 | relu | 0.0001 | 0.9298 | 0.9368 | 0.9298 | 0.9280 | 0.503 |
| ANN_logistic_LR_0.0001 | logistic | 0.0001 | 0.6316 | 0.3989 | 0.6316 | 0.4890 | 0.303 |
| ANN_logistic_LR_0.001 | logistic | 0.001 | 0.6316 | 0.3989 | 0.6316 | 0.4890 | 0.403 |

## 8. Best-performing experiment

The best experiment is selected using the highest weighted F1-score.

- **Experiment:** ANN_tanh_LR_0.001
- **Activation function:** tanh
- **Learning rate:** 0.001
- **Accuracy:** 0.9825
- **Weighted precision:** 0.9829
- **Weighted recall:** 0.9825
- **Weighted F1-score:** 0.9824
- **Execution time:** 0.403 seconds

A higher weighted F1-score indicates a better balance between
precision and recall across the classes. The best result is based on
this test-set comparison; it is not a guarantee of performance on
new external data.

## Best model classification report


Best model: ANN_tanh_LR_0.001

Seven hidden layers: (128, 64, 64, 32, 32, 16, 8)

              precision    recall  f1-score   support

           B       0.97      1.00      0.99        72
           M       1.00      0.95      0.98        42

    accuracy                           0.98       114
   macro avg       0.99      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
```


## 9. Output files

The ANN experiment script generates:

- `tsne_visualization.png` — two-dimensional t-SNE plot.
- `confusion_matrix_*.png` — confusion matrix for each experiment.
- `ANN_Results.csv` — comparison of all nine experiments, including
  evaluation and resource measurements.
- `Best_Model_Report.txt` — detailed classification report for the
  best model.
- `README.md` — this project documentation.

## 10. How to run

1. Open the ANN Python script in Spyder.
2. Run the script.
3. Select the dataset file when the file-selection window appears.
4. Wait for all nine experiments to complete.
5. The output files are saved in the `ANN_Assignment_Results` folder.
6. Run this README generator and select that results folder to create
   or update this `README.md`.

## 11. Notes and limitations

- The results depend on the selected dataset and the computer used.
- t-SNE is a visualization method; apparent cluster separation does
  not by itself establish classification performance.
- A deep network with seven hidden layers may be complex for a small
  dataset and may not converge within the configured iteration limit.
- Results should be reported exactly as produced by the experiment.
- For a stronger scientific evaluation, consider repeated
  stratified cross-validation and an independent external test set.


## 12. SVM vs Decision Tree vs ANN Classification model comparison

The best model is

- **Model:** ANN_tanh_LR_0.001
- **Activation function:** tanh
- **Learning rate:** 0.001
- **Accuracy:** 0.9825
- **Weighted precision:** 0.9829
- **Weighted recall:** 0.9825
- **Weighted F1-score:** 0.9824
- **Execution time:** 0.403 seconds