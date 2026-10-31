# Artificial Neural Network (ANN) for Dry Bean Classification

## Project Overview

This project implements an Artificial Neural Network (ANN) using Python and Scikit-learn to classify seven varieties of dry beans based on their morphological features.

The main objective is to investigate how the number of hidden layers, activation functions and learning rates influence ANN classification performance.

A total of 81 experiments were conducted to identify the best-performing ANN configuration.

## Dataset Description

The Dry Bean dataset contains 13,611 observations and 17 attributes.

- **Input features:** 16 numerical features
- **Target variable:** Class
- **Number of classes:** 7

### Bean Classes

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

### Input Features

Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3 and ShapeFactor4.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- GitHub

## Methodology

The following steps were performed:

1. Dataset loading and exploratory data analysis.
2. Missing-value inspection.
3. Separation of input features and target variable.
4. Target encoding using LabelEncoder.
5. Splitting the dataset into 80% training and 20% testing.
6. Feature standardization using StandardScaler.
7. ANN model development using MLPClassifier.
8. Performance evaluation using Accuracy, Precision, Recall and F1-score.
9. Comparison of different ANN configurations.
10. Identification and evaluation of the best-performing ANN model.

## ANN Experimental Parameters

A total of 81 experiments were performed by varying the following parameters:

| Parameter | Tested Values |
|---|---|
| Hidden layers | 2 to 10 |
| Neurons per hidden layer | 32 |
| Activation functions | ReLU, Logistic, Tanh |
| Learning rates | 0.0001, 0.001, 0.01 |
| Optimizer | Adam |
| Maximum iterations | 200 |
| Early stopping | Enabled |

## Best ANN Configuration

The best individual configuration was identified using Macro F1-score.

| Parameter | Best Configuration |
|---|---|
| Hidden layers | 4 |
| Neurons per layer | 32 |
| Activation function | ReLU |
| Learning rate | 0.01 |
| Optimizer | Adam |

## Final ANN Model Performance

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 92.80% |
| Macro Precision | 94.05% |
| Macro Recall | 93.94% |
| Macro F1-score | 93.98% |
| Training time | 2.42 seconds |

## Activation Function Comparison

ReLU achieved the highest average performance among the three tested activation functions.

| Activation | Average Accuracy | Average Macro F1 |
|---|---:|---:|
| Logistic | 57.73% | 46.96% |
| ReLU | 92.21% | 93.39% |
| Tanh | 92.03% | 93.14% |

## Learning Rate Comparison

The learning rate of 0.01 achieved the highest average Macro F1-score.

| Learning Rate | Average Accuracy | Average Macro F1 |
|---|---:|---:|
| 0.0001 | 72.51% | 67.38% |
| 0.001 | 84.76% | 82.64% |
| 0.01 | 84.71% | 83.47% |

## Hidden Layer Comparison

The architecture with two hidden layers achieved the highest average performance across its nine experiments.

However, the best individual configuration was the four-hidden-layer ReLU model.

## ANN vs SVM vs Decision Tree

| Model | Configuration | Accuracy | F1-score |
|---|---|---:|---:|
| SVM | Linear | 92.10% | 92.10% |
| SVM | Polynomial | 91.40% | 91.55% |
| SVM | RBF | 92.10% | 92.11% |
| Decision Tree | Depth 5 | 87.49% | 87.55% |
| ANN | 4 Layers - ReLU | 92.80% | 93.98% |

The ANN achieved the highest recorded accuracy and F1-score among these configurations. This comparison should be interpreted with the limitation that identical train/test splits and evaluation procedures have not been independently established for all models.

## Visualizations

The project includes:

- Nine ANN F1-score comparison graphs for hidden-layer counts from 2 to 10.
- Final ANN confusion matrix.
- ANN, SVM and Decision Tree performance comparison graph.

All visualizations are available in the `figures/` directory.

## Project Structure

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
│   └── ANN comparison graphs and confusion matrix
│
└── README.md
```

## How to Run the Project

1. Clone or download this repository.
2. Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

3. Open Jupyter Notebook.
4. Navigate to the `notebook/` directory.
5. Open `dry_bean_ann_classification.ipynb`.
6. Execute the notebook cells sequentially.

## Conclusion

The experiment demonstrates that Artificial Neural Networks can effectively classify seven varieties of dry beans using morphological features.

Among the 81 tested configurations, the four-hidden-layer ANN with 32 neurons per layer, ReLU activation and a learning rate of 0.01 achieved the highest Macro F1-score of 93.98%.

The results also demonstrate that increasing the number of hidden layers does not necessarily improve classification performance.

## Author

**Vyshnavi Perumal**

## Project Category

Artificial Intelligence | Machine Learning | Deep Learning | Multiclass Classification