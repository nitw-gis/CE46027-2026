# Artificial Neural Network for California Housing Price Prediction

## 1. Project Overview

This project develops an Artificial Neural Network (ANN) model to predict median house values using the California Housing dataset.

The study investigates how different network architectures, activation functions, and learning rates influence ANN performance in a regression problem.

A total of 81 ANN experiments were conducted using Scikit-learn.

## 2. Objectives

- Develop an ANN model for California housing price prediction.
- Examine the effect of different numbers of hidden layers.
- Compare ReLU, Logistic, and Tanh activation functions.
- Investigate the influence of different learning rates.
- Evaluate model performance using R², MAE, and RMSE.
- Identify the best-performing ANN configuration.

## 3. Dataset

- **Dataset:** California Housing
- **Total records:** 20,640
- **Input features:** 9
- **Target variable:** `median_house_value`
- **Problem type:** Regression

The input features include geographical coordinates, housing age, rooms, bedrooms, population, households, median income, and ocean proximity.

## 4. Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- GitHub

## 5. Methodology

1. Dataset loading and exploration.
2. Missing-value identification and treatment.
3. Separation of input features and target variable.
4. Training and testing split (80:20).
5. Numerical feature standardization.
6. Categorical feature encoding using One-Hot Encoding.
7. Target variable standardization during model training.
8. ANN model development using `MLPRegressor`.
9. Performance evaluation and comparison.

## 6. Experimental Design

| Parameter | Configurations |
|---|---|
| Hidden layers | 2 to 10 |
| Neurons per hidden layer | 32 |
| Activation functions | ReLU, Logistic, Tanh |
| Learning rates | 0.0001, 0.001, 0.01 |
| Optimizer | Adam |
| Maximum iterations | 200 |
| Early stopping | Enabled |
| Total experiments | 81 |

## 7. Best ANN Configuration

| Parameter | Best configuration |
|---|---|
| Hidden layers | 6 |
| Neurons per layer | 32 |
| Activation function | Tanh |
| Learning rate | 0.001 |
| R² | 0.803074 |
| MAE | 33,923.78 |
| RMSE | 50,799.01 |

## 8. Model Performance Comparison

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Multiple Linear Regression | 0.613866 | 51,810.48 | 71,133.17 |
| KNN Regression (K=7) | 0.709497 | 41,661.51 | 61,699.10 |
| Initial ANN | 0.770674 | 37,653.92 | 54,818.84 |
| Best ANN | 0.803074 | 33,923.78 | 50,799.01 |

## 9. Results and Findings

- The best-performing ANN configuration used six hidden layers with 32 neurons per layer.
- Tanh activation with a learning rate of 0.001 achieved the highest R² score.
- Increasing the number of hidden layers did not consistently improve performance.
- The best ANN configuration outperformed the previously developed MLR and KNN regression models on the recorded evaluation metrics.
- Nine comparison graphs were generated to investigate the effect of network architecture, activation functions, and learning rates.

## 10. Project Structure

```text
california-housing-ann/
│
├── data/
│   └── housing.csv
│
├── notebook/
│   └── california_ann.ipynb
│
├── results/
│   ├── ANN_Experiment_Results.csv
│   └── ANN_Top_10_Models.csv
│
├── figures/
│   ├── ANN_2_Hidden_Layers.png
│   ├── ANN_3_Hidden_Layers.png
│   ├── ANN_4_Hidden_Layers.png
│   ├── ANN_5_Hidden_Layers.png
│   ├── ANN_6_Hidden_Layers.png
│   ├── ANN_7_Hidden_Layers.png
│   ├── ANN_8_Hidden_Layers.png
│   ├── ANN_9_Hidden_Layers.png
│   ├── ANN_10_Hidden_Layers.png
│   ├── Model_Comparison_R2.png
│   ├── Model_Comparison_MAE.png
│   └── Model_Comparison_RMSE.png
│
└── README.md
```

## 11. Conclusion

The experimental study demonstrates that ANN performance depends on network architecture, activation functions, and learning rates. Among the 81 configurations tested, the six-hidden-layer Tanh model with a learning rate of 0.001 achieved the best predictive performance, with an R² score of 0.803074.

## Author

**Vyshnavi Perumal**