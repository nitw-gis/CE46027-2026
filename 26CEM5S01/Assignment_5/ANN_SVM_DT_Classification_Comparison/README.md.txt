# Artificial Neural Network Classification and Comparative Analysis of Machine Learning Classifiers

## Abstract

This project investigates the application of Artificial Neural Networks (ANN) for classification and compares their performance with Support Vector Machine (SVM) and Decision Tree (DT) algorithms. The experiments are conducted using the `Dry_Bean_Dataset`, which contains 13,611 samples, 16 numerical input features and seven target classes.

The first part of the project focuses on ANN classification. A total of 63 ANN configurations are evaluated by varying the number of hidden layers, activation functions and learning rates. The best configuration is selected using validation performance and subsequently evaluated on an independent testing dataset.

In the second part, the selected ANN model is compared with SVM and Decision Tree classifiers using the same data partitions. The models are evaluated using accuracy, precision, recall, F1-score and computational time. The results indicate that ANN achieves the highest overall ranking based on accuracy and weighted evaluation metrics, while SVM obtains a slightly higher macro F1-score. Decision Tree demonstrates the lowest computational time.

---

# 1. Introduction

Machine learning classification techniques are widely used to identify patterns in data and assign observations to predefined categories. Artificial Neural Networks, Support Vector Machines and Decision Trees are commonly used supervised learning algorithms for classification problems.

This project is divided into two major stages.

**Stage 1 – ANN Classification:** An Artificial Neural Network classifier is developed using the `Dry_Bean_Dataset`. Different network architectures and training parameters are investigated to identify the best-performing ANN configuration.

**Stage 2 – Comparative Analysis:** The selected ANN model is compared with SVM and Decision Tree classifiers. Their predictive performance and computational requirements are evaluated to determine their relative effectiveness.

The main objectives of this project are:

- To understand and implement ANN-based classification.
- To investigate the effect of different ANN architectures and training parameters.
- To identify the best-performing ANN configuration using validation results.
- To evaluate the selected ANN model using an independent testing dataset.
- To compare ANN, SVM and Decision Tree classifiers using common evaluation criteria.
- To identify the strengths and limitations of each classification algorithm.

# 2. Dataset Description

## 2.1 Dataset Overview

The **Dry_Bean_Dataset** is used for both the ANN classification experiment and the comparative analysis.

**Dataset file:** `Dry_Bean_Dataset.xlsx`

The dataset contains physical measurements of dry bean samples. These measurements are used to classify each sample into one of seven dry bean categories.

| Attribute | Description |
|---|---|
| Dataset name | Dry_Bean_Dataset |
| File format | Excel (.xlsx) |
| Total samples | 13,611 |
| Number of input features | 16 |
| Target variable | Class |
| Number of target classes | 7 |
| Missing feature values | 0 |
| Duplicate rows | 68 |

## 2.2 Input Features

The dataset consists of 16 numerical features describing the physical characteristics of the dry bean samples.

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

## 2.3 Target Classes

The target variable is `Class`, which represents the category of each dry bean sample.

The seven target classes are:

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

The classification objective is to predict the correct dry bean category using the 16 numerical input features.

# 3. ANN Classification: Concept and Architecture

## 3.1 Introduction to Artificial Neural Networks

An Artificial Neural Network is a supervised machine learning model inspired by the structure and functioning of biological neural networks. It consists of interconnected processing units called neurons, organized into layers.

An ANN learns the relationship between input features and target classes by adjusting the weights and biases of connections between neurons during training.

For classification problems, the network learns patterns in the input data and uses these learned relationships to predict the class of previously unseen observations.

## 3.2 ANN Architecture

The ANN classifier consists of three main types of layers.

**Input Layer:** Receives the 16 numerical features describing each dry bean sample.

**Hidden Layers:** Learn patterns and relationships between input features through weighted connections and activation functions. The number of hidden layers was varied during the experiments.

**Output Layer:** Produces the classification output corresponding to the seven dry bean categories.

## 3.3 Working Principle of ANN

The ANN classification process involves the following operations:

1. The input features are supplied to the network.
2. Each neuron calculates a weighted sum of its inputs and adds a bias.
3. An activation function transforms the calculated value.
4. Information is passed through the hidden layers.
5. The output layer generates predictions for the target classes.
6. During training, the predicted outputs are compared with the actual class labels.
7. The network adjusts its weights and biases to reduce classification error.
8. The process continues until the stopping criteria are reached.

Through repeated training, the ANN learns patterns that help it classify previously unseen dry bean samples.

# 4. ANN Classification: Experimental Methodology

## 4.1 Data Loading and Feature Separation

The `Dry_Bean_Dataset.xlsx` file was loaded using Pandas.

The input features were separated from the target variable `Class`. The 16 numerical features were used as independent variables, while `Class` was used as the dependent variable.

## 4.2 Target Encoding

The target variable contains seven categorical class labels.

`LabelEncoder` was used to convert these categorical labels into numerical representations so that they could be processed by the ANN classifier.

## 4.3 Missing Value Handling

A median imputation strategy was implemented using Scikit-learn's `SimpleImputer`.

Although the dataset contained no missing feature values, the imputation step was retained as part of the preprocessing workflow.

## 4.4 Feature Standardization

`StandardScaler` was used to standardize the numerical input features.

Standardization transforms the features to have approximately zero mean and unit variance. This helps prevent features with larger numerical ranges from dominating the learning process.

The imputer and scaler were fitted using the training data and subsequently applied to the validation and testing data.

## 4.5 Training, Validation and Testing Data Split

The dataset was divided into training, validation and testing subsets using stratified sampling.

The splitting procedure involved two stages:

- First split: 80% training-validation data and 20% testing data.
- Second split: 80% training data and 20% validation data from the training-validation subset.

The following parameters were used:

- Random state: `42`
- Stratification: Target class

| Dataset subset | Number of samples | Percentage |
|---|---:|---:|
| Training | 8,710 | 64% |
| Validation | 2,178 | 16% |
| Testing | 2,723 | 20% |
| **Total** | **13,611** | **100%** |

Stratified sampling maintains the approximate distribution of the seven target classes across the three subsets.

The training data is used to fit the ANN, the validation data is used to select the best ANN configuration, and the testing data is used for final performance evaluation.

## 4.6 ANN Configuration and Hyperparameter Experiments

A total of **63 ANN configurations** were evaluated to investigate the effect of different network architectures and training parameters.

The experiments varied the following parameters:

| Experimental parameter | Values tested |
|---|---|
| Number of hidden layers | 1, 2, 3, 4, 5, 6, 7 |
| Neurons per hidden layer | 16 |
| Activation functions | ReLU, Sigmoid, Tanh |
| Learning rates | 0.001, 0.005, 0.01 |

The total number of configurations was:

7 hidden-layer configurations × 3 activation functions × 3 learning rates = **63 experiments**.

The ANN classifier was implemented using Scikit-learn's `MLPClassifier`.

The remaining training parameters were:

| Parameter | Value |
|---|---|
| Solver | Adam |
| Maximum iterations | 500 |
| Random state | 42 |
| Early stopping | Enabled |
| Validation fraction | 0.10 |
| No-improvement iterations | 25 |
| Tolerance | 0.0001 |

## 4.7 Best ANN Configuration Selection

The 63 ANN configurations were evaluated using the validation dataset.

The best configuration was selected by prioritizing the following metrics:

1. Weighted F1-score
2. Accuracy
3. Weighted recall
4. Weighted precision

The configuration with the best validation performance was selected for final testing.

### Selected ANN Configuration

| Parameter | Best value |
|---|---|
| Number of hidden layers | **2** |
| Neurons per hidden layer | **16** |
| Activation function | **Sigmoid** |
| Learning rate | **0.001** |
| Solver | Adam |
| Iterations completed | 376 |

The selected ANN model was trained with two hidden layers containing 16 neurons each, using Sigmoid activation and a learning rate of 0.001.

# 5. ANN Classification: Evaluation and Results

## 5.1 Evaluation Metrics

The ANN classifier was evaluated using the following classification metrics.

**Accuracy:** Measures the proportion of correctly classified samples out of all tested samples.

**Precision:** Measures the proportion of samples predicted as a particular class that actually belong to that class.

**Recall:** Measures the proportion of actual samples belonging to a class that are correctly identified by the model.

**F1-score:** Represents the harmonic mean of precision and recall, providing a combined measure of the two metrics.

### Weighted Average

Weighted precision, recall and F1-score calculate the metric for each class and combine the results according to the number of samples in each class. This accounts for differences in class frequencies.

### Macro Average

Macro precision, recall and F1-score calculate the metric for each class and then take the unweighted average. Each class receives equal importance, irrespective of its number of samples.

Both weighted and macro metrics are useful because the Dry_Bean_Dataset does not contain equal numbers of samples in all seven classes.

## 5.2 ANN Validation Results

The selected ANN configuration achieved the following validation results:

| Evaluation metric | Validation result |
|---|---:|
| Accuracy | 0.940771 |
| Weighted Precision | 0.941095 |
| Weighted Recall | 0.940771 |
| Weighted F1-score | 0.940899 |
| Macro Precision | 0.951175 |
| Macro Recall | 0.948449 |
| Macro F1-score | 0.949781 |

The selected ANN achieved a validation accuracy of approximately **94.08%** and a validation weighted F1-score of approximately **94.09%**.

## 5.3 ANN Testing Results

After selecting the best configuration, its performance was evaluated on the independent testing dataset containing 2,723 samples.

| Evaluation metric | Testing result |
|---|---:|
| Accuracy | **0.924348** |
| Weighted Precision | 0.924502 |
| Weighted Recall | 0.924348 |
| Weighted F1-score | 0.924365 |
| Macro Precision | 0.935735 |
| Macro Recall | 0.933823 |
| Macro F1-score | 0.934712 |

The ANN achieved a test accuracy of approximately **92.43%**, indicating that it correctly classified most of the dry bean samples in the testing dataset.

The weighted F1-score of 0.924365 indicates a strong balance between precision and recall when class frequencies are considered.

The macro F1-score of 0.934712 provides an additional view of performance by giving equal importance to all seven classes.

## 5.4 Best Performing ANN Model: Summary

Based on the evaluation of 63 different ANN configurations, the model with two hidden layers, 16 neurons per layer, Sigmoid activation and a learning rate of 0.001 was identified as the best-performing ANN configuration.

Its final performance is summarized below.

| Parameter | Selected value |
|---|---|
| Hidden layers | **2** |
| Neurons in each hidden layer | **16** |
| Activation function | **Sigmoid** |
| Learning rate | **0.001** |
| Optimizer | Adam |
| Maximum iterations | 500 |
| Actual iterations completed | 376 |
| Early stopping | Enabled |

| Evaluation metric | Validation | Testing |
|---|---:|---:|
| Accuracy | 0.940771 | 0.924348 |
| Weighted Precision | 0.941095 | 0.924502 |
| Weighted Recall | 0.940771 | 0.924348 |
| Weighted F1-score | 0.940899 | 0.924365 |
| Macro Precision | 0.951175 | 0.935735 |
| Macro Recall | 0.948449 | 0.933823 |
| Macro F1-score | 0.949781 | 0.934712 |

The selected ANN achieved a validation accuracy of approximately **94.08%** and a testing accuracy of approximately **92.43%**.

The testing results provide the final estimate of the model's performance on unseen observations. This selected ANN configuration was subsequently used in the comparative analysis with SVM and Decision Tree.

## 5.5 Explanation of ANN Classification Plots

The plots generated during the ANN experiment are used to understand the performance of the different configurations and evaluate the selected model.

### ANN Configuration Performance Plot

This plot compares the validation performance of the different ANN configurations.

It helps identify how changes in the number of hidden layers, activation functions and learning rates affect classification performance.

### ANN Confusion Matrix

The confusion matrix compares the actual class labels with the class labels predicted by the ANN.

- Diagonal values represent correctly classified samples.
- Off-diagonal values represent incorrectly classified samples.
- Rows and columns show the actual and predicted class distributions, respectively, according to the plot's axis labels.

The confusion matrix helps identify which dry bean categories are classified correctly and which categories are more frequently confused with other classes.

### Other Evaluation Plots

Where generated, metric comparison plots provide a visual representation of accuracy, precision, recall and F1-score.

These plots make it easier to understand the model's performance and identify differences between evaluation metrics.

## 5.6 Explanation of ANN Output Tables

The ANN experiment generates tables to record the configurations, model performance and data partitions.

**ANN Experiment Results Table – `ann_63_experiments.csv`**

Records the results of all 63 ANN configurations. It is used to compare the validation performance of different network architectures, activation functions and learning rates.

**Model Performance Table**

Records the selected ANN model's evaluation metrics, including accuracy, precision, recall and F1-score.

**Training, Validation and Testing Indices**

The following files store the row indices belonging to each data partition:

- `training_indices.csv`
- `validation_indices.csv`
- `testing_indices.csv`

These files support reproducibility by recording which observations were assigned to each subset.

**Classification Predictions Table**

Records the actual and predicted class labels for the testing samples. It can be used to identify individual correct and incorrect predictions.

---

# 6. Comparative Analysis of ANN, SVM and Decision Tree

## 6.1 Introduction to Model Comparison

After developing and evaluating the ANN classification model, its performance was compared with two additional machine learning classifiers:

1. Artificial Neural Network (ANN)
2. Support Vector Machine (SVM)
3. Decision Tree (DT)

The objective is to identify differences in classification performance and computational requirements.

The same `Dry_Bean_Dataset` and the same training, validation and testing partitions were used for all three models. This allows their final predictions to be compared on the same held-out testing observations.

## 6.2 Support Vector Machine (SVM)

Support Vector Machine is a supervised machine learning algorithm that identifies a decision boundary separating different classes.

The RBF kernel allows the SVM to model nonlinear decision boundaries.

The following configuration was used:

| Parameter | Value |
|---|---|
| Kernel | RBF |
| C | 1.0 |
| Gamma | scale |

Feature standardization was applied before training the SVM classifier.

## 6.3 Decision Tree (DT)

A Decision Tree is a supervised classification algorithm that creates a tree-like structure of decision rules.

It repeatedly divides the dataset into smaller subsets based on feature conditions. The final leaf nodes represent the predicted classes.

The following configuration was used:

| Parameter | Value |
|---|---|
| Criterion | Gini |
| Random state | 42 |

The Decision Tree was trained using median-imputed features without standardization.

## 6.4 Comparative Experimental Methodology

The comparison experiment followed the same general data preparation and partitioning procedure described in the ANN classification section.

### Data Preparation

The target column `Class` was encoded using `LabelEncoder`.

Missing feature values were handled using median imputation.

For ANN and SVM, the input features were standardized using `StandardScaler`. The Decision Tree used the imputed features without standardization.

### Common Data Partitions

All three models were evaluated using the same data partitions:

| Dataset subset | Number of samples |
|---|---:|
| Training | 8,710 |
| Validation | 2,178 |
| Testing | 2,723 |

The ANN used the validation dataset for selecting its best configuration.

SVM and Decision Tree used their specified fixed configurations and were trained on the training subset. Their final evaluation was performed on the same testing subset.

### Model Training and Testing

The three classifiers were trained separately.

After training, each model predicted the classes of the same 2,723 testing observations.

The predicted class labels were compared with the actual labels to calculate the evaluation metrics.

### Performance Evaluation

The comparison considered:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Macro precision
- Macro recall
- Macro F1-score
- Training time
- Prediction time

The models were ranked using accuracy, weighted precision, weighted recall and weighted F1-score.

# 7. Comparative Results and Discussion

## 7.1 Weighted Performance Metrics

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| ANN | **0.924348** | **0.924502** | **0.924348** | **0.924365** |
| SVM | 0.923614 | 0.923990 | 0.923614 | 0.923648 |
| Decision Tree | 0.889460 | 0.889198 | 0.889460 | 0.889246 |

ANN achieved the highest accuracy and weighted-average precision, recall and F1-score.

SVM performed very closely to ANN, while Decision Tree achieved lower values across these weighted metrics.

## 7.2 Macro Performance Metrics

| Model | Macro Precision | Macro Recall | Macro F1-score |
|---|---:|---:|---:|
| ANN | 0.935735 | 0.933823 | 0.934712 |
| SVM | **0.937400** | 0.933708 | **0.935342** |
| Decision Tree | 0.904941 | **0.905680** | 0.905231 |

SVM achieved the highest macro precision and macro F1-score. This indicates slightly stronger average performance across the seven classes when each class is given equal importance.

ANN achieved the highest macro recall by a small margin over SVM.

## 7.3 Computational Time Comparison

| Model | Training Time (seconds) | Prediction Time (seconds) |
|---|---:|---:|
| ANN | 15.349259 | 0.010985 |
| SVM | 0.354129 | 0.490182 |
| Decision Tree | **0.211726** | **0.002231** |

Decision Tree required the least time for training and prediction.

SVM had a short training time but required more time than ANN and Decision Tree for test prediction.

The reported ANN training time represents the final selected ANN configuration's training time. It does not include the time required to evaluate all 63 ANN configurations.

## 7.4 Overall Model Ranking

The models were ranked based on accuracy, weighted precision, weighted recall and weighted F1-score.

| Model | Accuracy Rank | Precision Rank | Recall Rank | F1 Rank | Average Rank | Overall Rank |
|---|---:|---:|---:|---:|---:|---:|
| ANN | 1 | 1 | 1 | 1 | 1.00 | **1** |
| SVM | 2 | 2 | 2 | 2 | 2.00 | **2** |
| Decision Tree | 3 | 3 | 3 | 3 | 3.00 | **3** |

ANN achieved the first overall rank, followed by SVM and Decision Tree.

The ranking is based on weighted metrics and accuracy; it does not mean ANN achieved the highest value for every metric. SVM achieved a slightly higher macro F1-score.

## 7.5 Explanation of Comparative Plots

The comparison experiment generates several plots to visualize differences between the three classifiers.

**Accuracy Comparison – `comparison_accuracy.png`**

This plot compares the test accuracy of ANN, SVM and Decision Tree. ANN achieved the highest accuracy, followed closely by SVM. Decision Tree achieved the lowest accuracy.

**Weighted Precision Comparison – `comparison_precision_weighted.png`**

This plot compares weighted precision across the three models. ANN achieved the highest weighted precision, followed by SVM and Decision Tree.

**Weighted Recall Comparison – `comparison_recall_weighted.png`**

This plot compares the weighted recall values of the three classifiers. ANN achieved the highest weighted recall, followed by SVM and Decision Tree.

**Weighted F1-score Comparison – `comparison_f1_weighted.png`**

This plot compares the weighted F1-scores. ANN achieved the highest weighted F1-score, indicating the best balance of weighted precision and recall among the three models.

**Computational Time Comparison – `comparison_computational_time.png`**

This plot compares the computational time required for training and testing predictions. Decision Tree was the fastest model in both operations. SVM required the longest test prediction time.

The ANN training time shown represents only the selected configuration's training time and excludes the full 63-configuration search.

**Confusion Matrices**

The following confusion matrix plots were generated:

- `confusion_matrix_ann.png`
- `confusion_matrix_svm.png`
- `confusion_matrix_decision_tree.png`

Each confusion matrix compares the actual and predicted class labels for its respective classifier.

The diagonal cells represent correct classifications, while the off-diagonal cells indicate misclassifications. Comparing these matrices helps identify differences in class-wise prediction performance and the categories that are more frequently confused.

## 7.6 Explanation of Comparative Output Tables

**ANN Experiment Results – `ann_63_experiments.csv`**

Contains the validation results for all 63 ANN configurations. It records the performance of the different combinations of hidden layers, activation functions and learning rates.

**Classification Model Comparison – `classification_model_comparison.csv`**

Contains the test performance metrics of ANN, SVM and Decision Tree. It allows direct numerical comparison of the models using accuracy, precision, recall and F1-score.

**Classification Performance Ranking – `classification_performance_ranking.csv`**

Contains the individual metric ranks, average rank and overall rank of the three classifiers. It identifies ANN as the first-ranked model according to the selected ranking criteria.

**Per-Class Metrics – `classification_per_class_metrics.csv`**

Contains class-wise evaluation metrics for the classification models. It provides a more detailed understanding of how each model performs on the seven dry bean categories.

**Test Predictions – `classification_test_predictions.csv`**

Contains the actual and predicted class labels for the testing observations. It can be used to examine individual predictions and identify correctly classified and misclassified samples.

**Data Split Indices**

The following files record the indices used for the three data partitions:

- `training_indices.csv`
- `validation_indices.csv`
- `testing_indices.csv`

These files support reproducibility and help ensure that the same observations are used for evaluating the models.

## 7.7 Discussion

The experimental comparison shows that ANN and SVM achieved very similar classification accuracy on the `Dry_Bean_Dataset`.

ANN obtained a test accuracy of 92.43%, while SVM obtained 92.36%. ANN also achieved the highest weighted precision, recall and F1-score.

However, SVM obtained a slightly higher macro F1-score than ANN. This suggests that SVM performed marginally better when the seven classes were given equal importance.

Decision Tree achieved a lower accuracy of 88.95%, but it required the least training and prediction time.

The results demonstrate that the choice of classifier depends on the evaluation objective:

- **ANN:** Best overall ranking and highest test accuracy and weighted evaluation metrics.
- **SVM:** Competitive classification performance and highest macro F1-score.
- **Decision Tree:** Lowest classification performance among the three models, but fastest training and prediction.

# 8. Conclusion

This project implemented ANN classification using the `Dry_Bean_Dataset` and evaluated different ANN architectures through 63 experimental configurations.

The best ANN configuration consisted of two hidden layers with 16 neurons per layer, Sigmoid activation and a learning rate of 0.001. It achieved a validation accuracy of approximately 94.08% and a test accuracy of approximately 92.43%.

The selected ANN was subsequently compared with SVM and Decision Tree classifiers using the same held-out testing dataset.

ANN achieved the highest overall ranking based on accuracy and weighted evaluation metrics. SVM performed very closely to ANN and achieved the highest macro F1-score. Decision Tree achieved the lowest classification accuracy but required the least computational time.

Therefore, ANN was the best-performing model according to the selected overall ranking criteria, while SVM and Decision Tree provided useful alternatives based on class-wise performance and computational efficiency.

# 9. Software and Libraries Used

The project was implemented using the following tools and libraries:

| Tool or library | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing, ANN, SVM, Decision Tree and evaluation metrics |
| Matplotlib | Plot generation |
| Seaborn | Statistical visualizations |
| OpenPyXL | Reading Excel datasets |

# 10. Reproducibility Notes

- The experiments use the `Dry_Bean_Dataset.xlsx` dataset.
- Random state `42` was used for the stratified data partitioning.
- The same training, validation and testing indices were used for all three classifiers.
- The ANN configuration was selected using validation performance.
- The final performance metrics were calculated using the held-out testing dataset.
- The saved CSV files contain model results, predictions, rankings and data partition indices.
- The ANN, SVM and Decision Tree configurations are documented in this README.
- The ANN computational time reported in the comparison represents the selected configuration's training time, not the total time required for the 63 ANN experiments.