# Dry Bean Classification Using Artificial Neural Networks (ANN)

## 1. Project Overview

This project implements an Artificial Neural Network (ANN) model to classify dry bean varieties based on their physical characteristics.

The project uses the Dry Bean Dataset, which contains 16 numerical features describing the dimensions, shape, and other geometric properties of dry beans. The ANN model learns patterns from these features to classify the beans into seven different varieties.

A systematic experiment was conducted using different hidden-layer configurations, activation functions, and learning rates to understand their influence on classification performance.

## 2. Objectives

- Understand the fundamentals of Artificial Neural Networks.
- Perform multiclass classification using ANN.
- Explore the influence of different activation functions.
- Investigate the effect of hidden-layer architecture.
- Evaluate different learning rates.
- Conduct 81 ANN experiments using different parameter combinations.
- Compare ANN performance with previously developed SVM and Decision Tree models.

## 3. Dataset Description

**Dataset:** Dry Bean Dataset

**Number of observations:** 13,611

**Number of input features:** 16

**Number of target classes:** 7

The dataset contains measurements of dry beans obtained through computer vision techniques.

### Input Features

The 16 numerical features include:

- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Aspect Ratio
- Eccentricity
- Convex Area
- Equivalent Diameter
- Extent
- Solidity
- Roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4

### Target Classes

The target variable is `Class`, containing seven dry bean varieties:

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

## 4. Technologies and Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- GitHub

## 5. Methodology

The following workflow was implemented:

### 5.1 Data Loading and Exploration

- Imported the Dry Bean Dataset using Pandas.
- Examined the dataset dimensions and data types.
- Checked for missing values.
- Analysed the distribution of the seven target classes.

### 5.2 Data Preparation

- Separated the input features and target variable.
- Converted categorical target labels into numerical labels using LabelEncoder.
- Divided the dataset into training and testing subsets using an 80:20 ratio.
- Applied stratified sampling to maintain class proportions.

### 5.3 Feature Scaling

StandardScaler was applied to standardize the input features.

The scaler was fitted using only the training data and subsequently applied to the testing data to avoid data leakage.

### 5.4 ANN Model Development

The ANN classification model was implemented using Scikit-learn's `MLPClassifier`.

The model uses:
- Adam optimizer
- 32 neurons per hidden layer
- Maximum iterations: 200
- Early stopping enabled
- Validation fraction: 0.10

## 6. ANN Experimental Design

A total of **81 experiments** were conducted to investigate the influence of different ANN configurations.

### Experimental Parameters

| Parameter | Values |
|---|---|
| Hidden layers | 2 to 10 |
| Neurons per hidden layer | 32 |
| Activation functions | ReLU, Logistic, Tanh |
| Learning rates | 0.0001, 0.001, 0.01 |
| Optimizer | Adam |
| Maximum iterations | 200 |
| Early stopping | Enabled |

### Number of Experiments

The total number of experiments was calculated as:

9 hidden-layer configurations × 3 activation functions × 3 learning rates = **81 experiments**

### Activation Functions

Three activation functions were investigated:

- **ReLU:** Rectified Linear Unit, which introduces non-linearity and is commonly used in hidden layers.
- **Logistic:** Also known as the sigmoid activation function, which produces outputs between 0 and 1.
- **Tanh:** Hyperbolic tangent activation function, which produces outputs between -1 and 1.

The activation function was varied in the hidden layers. The multiclass output layer uses the softmax function.

## 7. Best ANN Model Configuration

The best individual configuration based on Macro F1-score was:

| Parameter | Best Configuration |
|---|---|
| Hidden layers | 4 |
| Neurons per layer | 32 |
| Activation function | ReLU |
| Learning rate | 0.01 |
| Optimizer | Adam |
| Training iterations | 39 |

## 8. Final ANN Model Performance

The selected ANN model achieved the following performance on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 92.80% |
| Macro Precision | 94.05% |
| Macro Recall | 93.94% |
| Macro F1-score | 93.98% |

The test dataset contained 2,723 observations, of which 2,527 were correctly classified.

### Classification Performance

The model performed well across all seven dry bean varieties.

- BOMBAY achieved 100% precision, recall, and F1-score on the test samples.
- HOROZ achieved an F1-score of 96%.
- CALI and SEKER achieved F1-scores of approximately 95%.
- SIRA had the lowest recall at approximately 85%.

The confusion matrix indicated that some SIRA samples were incorrectly classified as DERMASON.

## 9. Experimental Analysis

### 9.1 Activation Function Comparison

The average Macro F1-scores across the experiments were:

| Activation Function | Average Macro F1 |
|---|---:|
| ReLU | 0.9339 |
| Tanh | 0.9314 |
| Logistic | 0.4696 |

ReLU achieved the highest average Macro F1-score, followed closely by Tanh.

Logistic produced a lower average across all configurations, although some individual Logistic configurations performed well.

### 9.2 Learning Rate Comparison

| Learning Rate | Average Macro F1 |
|---|---:|
| 0.0001 | 0.6738 |
| 0.001 | 0.8264 |
| 0.01 | 0.8347 |

The learning rate of 0.01 achieved the highest average Macro F1-score and the lowest average training time among the three tested learning rates.

### 9.3 Hidden-Layer Comparison

The average Macro F1-score across different hidden-layer configurations was:

| Hidden Layers | Average Macro F1 |
|---|---:|
| 2 | 0.9331 |
| 3 | 0.8349 |
| 4 | 0.8373 |
| 5 | 0.8364 |
| 6 | 0.8328 |
| 7 | 0.7972 |
| 8 | 0.6527 |
| 9 | 0.6401 |
| 10 | 0.6399 |

The two-hidden-layer architecture achieved the highest average performance across its nine experimental combinations.

However, the best individual configuration was the four-hidden-layer ReLU model with a learning rate of 0.01.

This demonstrates that increasing the number of hidden layers does not necessarily improve ANN classification performance.

## 10. ANN, SVM and Decision Tree Comparison

The ANN model was compared with previously developed SVM and Decision Tree models.

| Model | Configuration | Accuracy | F1-score |
|---|---|---:|---:|
| ANN | 4 layers, ReLU, LR 0.01 | 0.9280 | 0.9398 |
| SVM | Linear | 0.9210 | 0.9210 |
| SVM | Polynomial | 0.9140 | 0.9155 |
| SVM | RBF | 0.9210 | 0.9211 |
| Decision Tree | Depth 5 | 0.8749 | 0.8755 |

The ANN achieved the highest recorded accuracy and F1-score among these configurations.

**Note:** These results are from previously developed models. The comparison should be interpreted cautiously because identical data splits, preprocessing procedures, and evaluation settings have not been independently verified for every model.

## 11. Evaluation Metrics

The following classification metrics were used:

- **Accuracy:** Measures the proportion of correctly classified observations.
- **Precision:** Measures how many predicted positive class observations were actually correct, calculated per class.
- **Recall:** Measures how many actual observations of a class were correctly identified.
- **F1-score:** Represents the harmonic mean of precision and recall.

Macro averaging was used for precision, recall, and F1-score so that each bean variety contributed equally to the overall metric.

## 12. Project Structure

```text
dry-bean-ann-classification/
│
├── data/
│   └── Dry_Bean_Dataset.xlsx
│
├── notebook/
│   └── dry_bean_ann_classification.ipynb
│
├── results/
│   ├── ANN_Classification_Experiment_Results.csv
│   ├── ANN_Activation_Comparison.csv
│   ├── ANN_Learning_Rate_Comparison.csv
│   ├── ANN_Hidden_Layer_Comparison.csv
│   ├── ANN_Final_Classification_Report.csv
│   ├── ANN_Final_Confusion_Matrix.csv
│   └── ANN_SVM_DT_Comparison.csv
│
├── figures/
│   ├── ANN_F1_Comparison_2_Hidden_Layers.png
│   ├── ANN_F1_Comparison_3_Hidden_Layers.png
│   ├── ANN_F1_Comparison_4_Hidden_Layers.png
│   ├── ANN_F1_Comparison_5_Hidden_Layers.png
│   ├── ANN_F1_Comparison_6_Hidden_Layers.png
│   ├── ANN_F1_Comparison_7_Hidden_Layers.png
│   ├── ANN_F1_Comparison_8_Hidden_Layers.png
│   ├── ANN_F1_Comparison_9_Hidden_Layers.png
│   ├── ANN_F1_Comparison_10_Hidden_Layers.png
│   ├── ANN_Final_Confusion_Matrix.png
│   └── ANN_SVM_DT_Performance_Comparison.png
│
├── README.md
└── requirements.txt
```

## 13. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/dry-bean-ann-classification.git
```

### Step 2: Navigate to the Project Directory

```bash
cd dry-bean-ann-classification
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Open Jupyter Notebook

```bash
jupyter notebook
```

Open `dry_bean_ann_classification.ipynb` from the `notebook` directory and execute the cells sequentially.

## 14. Limitations and Future Scope

- The model's performance may vary with different train-test splits and random seeds.
- The dataset has an uneven distribution among the seven classes.
- SIRA showed comparatively lower recall than the other classes.
- Selecting the best configuration using test-set F1-scores can introduce selection bias. A separate validation dataset or cross-validation would provide a more rigorous model-selection procedure.

Future improvements could include:
- Hyperparameter optimization.
- Cross-validation.
- Class-imbalance handling.
- Testing additional ANN architectures.
- Comparing ANN with other advanced classification algorithms.

## 15. Conclusion

This project demonstrates the application of Artificial Neural Networks for multiclass classification of dry bean varieties.

A total of 81 experiments were conducted by varying hidden-layer counts, activation functions, and learning rates.

The best individual ANN configuration consisted of four hidden layers with 32 neurons per layer, ReLU activation, and a learning rate of 0.01. It achieved 92.80% accuracy and a Macro F1-score of 93.98%.

The experimental analysis also demonstrated that deeper neural networks do not necessarily produce better results and that model performance depends on the combination of architectural and training parameters.

The project provided practical experience in ANN architecture design, multiclass classification, hyperparameter experimentation, model evaluation, and performance comparison.

## Author

**Vyshnavi Perumal**

Artificial Intelligence and Machine Learning | Geoinformatics
