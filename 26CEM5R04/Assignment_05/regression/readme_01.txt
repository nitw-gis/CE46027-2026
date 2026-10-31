 Building Energy Efficiency Prediction Using Machine Learning and Artificial Neural Networks

 1. Project Overview

Building energy efficiency is an important aspect of sustainable building design and energy management. Predicting the energy required for heating and cooling helps architects, engineers, and planners understand building performance and make informed design decisions.

This project applies three machine learning and deep learning approaches to predict building energy requirements:

- Multiple Linear Regression (MLR)
- K-Nearest Neighbors Regression (KNNR)
- Artificial Neural Network (ANN)

The project focuses on predicting **Heating Load** using eight building design features. ANN is further investigated by experimenting with different numbers of hidden layers, activation functions, and learning rates.

The final objective is to evaluate and compare the predictive performance of all three models and identify the most suitable model for the selected dataset.

---

2. Project Objectives

1. Understand the relationship between building design parameters and energy requirements.
2. Develop regression models to predict building Heating Load.
3. Implement Multiple Linear Regression and K-Nearest Neighbors Regression.
4. Develop an Artificial Neural Network for energy prediction.
5. Study the influence of hidden layers, activation functions, and learning rates on ANN performance.
6. Evaluate the models using R² Score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
7. Compare the performance of MLR, KNNR, and ANN.
8. Identify the best-performing model based on prediction accuracy and error metrics.

---

## 3. Dataset Description

**Dataset:** Energy Efficiency on Buildings

**Source:** Kaggle

**Dataset link:** https://www.kaggle.com/datasets/winternguyen/energy-efficiency-on-buildings

The dataset contains building design parameters and their corresponding heating and cooling energy requirements.

### Dataset characteristics

- Total observations: 768
- Total columns: 10
- Input features: 8
- Output variables: 2

### Input Features

| Feature | Description |
|---|---|
| Relative Compactness | Indicates the compactness of the building |
| Surface Area | Total external surface area |
| Wall Area | Total wall surface area |
| Roof Area | Total roof surface area |
| Overall Height | Overall building height |
| Orientation | Building orientation |
| Glazing Area | Percentage of glazed area |
| Glazing Area Distribution | Distribution of glazing across building surfaces |

### Target Variables

| Target | Description |
|---|---|
| Heating Load | Energy required to maintain indoor temperature during heating conditions |
| Cooling Load | Energy required to maintain indoor temperature during cooling conditions |

**Target selected for the three-model comparison:** Heating Load.

The ANN model and final three-model comparison in this project focus on Heating Load. The original MLR and KNNR comparison file also contains results for Cooling Load.

---

## 4. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Development environment |
| Pandas | Dataset handling and analysis |
| NumPy | Numerical operations |
| Matplotlib | Graph generation and visualization |
| Scikit-learn | Data preprocessing, splitting, and evaluation |
| TensorFlow / Keras | Artificial Neural Network development |
| KaggleHub | Dataset download |

---

## 5. Project Methodology

The overall workflow followed in this project is:

1. Dataset acquisition
2. Data loading and inspection
3. Selection of input features and target variable
4. Training and testing data separation
5. Data preprocessing
6. Development of MLR and KNNR models
7. Development of the ANN model
8. ANN parameter experimentation
9. Final ANN model development
10. Model evaluation
11. Comparative analysis
12. Result visualization and documentation

### 5.1 Data Preparation

The dataset was loaded into Python using Pandas.

The first eight columns were selected as input features, and Heating Load was selected as the prediction target.

The dataset was divided into:

- Training data: 80%
- Testing data: 20%

The `train_test_split()` function was used with `random_state=42` to maintain a reproducible split.

### 5.2 Feature Scaling

Standardization was applied to the ANN input features using `StandardScaler`.

The scaler was fitted using only the training data and then applied to the testing data.

This transformation helps place the input features on comparable scales and supports stable ANN training.

---

## 6. Machine Learning Models

### 6.1 Multiple Linear Regression (MLR)

Multiple Linear Regression is a supervised learning algorithm used to predict a continuous output based on multiple independent input variables.

In this project, the eight building design features are used to estimate Heating Load.

The general equation is:

\[
Y = b_0+b_1X_1+b_2X_2+\cdots+b_nX_n
\]

Where:

- Y = Predicted Heating Load
- b₀ = Intercept
- b₁, b₂, ..., bₙ = Regression coefficients
- X₁, X₂, ..., Xₙ = Input features

### 6.2 K-Nearest Neighbors Regression (KNNR)

KNN Regression predicts the output of a new observation by considering the target values of its nearest observations in the training dataset.

The algorithm calculates distances between observations and identifies the nearest neighbors.

The predicted value is generally calculated using the average of the target values of the selected neighbors.

KNNR is useful for identifying relationships that may not follow a simple linear pattern.

### 6.3 Artificial Neural Network (ANN)

An Artificial Neural Network is a deep learning approach inspired by the interconnected structure of biological neurons.

The ANN developed in this project consists of:

- Input layer: 8 neurons corresponding to the eight input features
- Hidden layers: Multiple layers containing 64 neurons each
- Output layer: 1 neuron for predicting Heating Load

The network uses the following components:

| Component | Configuration |
|---|---|
| Input features | 8 |
| Hidden layer neurons | 64 per layer |
| Output neurons | 1 |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Evaluation metric during training | Mean Absolute Error |
| Training batch size | 32 |

The ANN learns relationships between building design parameters and Heating Load by adjusting its weights and biases during training.

---

## 7. ANN Parameter Experiments

To understand how ANN architecture and training parameters influence prediction performance, three experiments were conducted.

### 7.1 Hidden Layer Experiment

The number of hidden layers was varied from 2 to 10.

- Number of configurations: 9
- Neurons per hidden layer: 64
- Activation function: ReLU
- Learning rate: 0.001

The objective was to examine how increasing network depth affects prediction accuracy and error.

**Best observed configuration:**

- Hidden layers: 5
- R² Score: 0.995004
- RMSE: 0.721617
- MAE: 0.534064

Graphs generated:

- Hidden Layers vs RMSE
- Hidden Layers vs MAE
- Hidden Layers vs R² Score

### 7.2 Activation Function Experiment

Three activation functions were tested:

1. ReLU
2. Sigmoid
3. Tanh

The network architecture was kept at two hidden layers, with 64 neurons per layer and a learning rate of 0.001.

**Experimental results:**

| Activation | R² Score | RMSE | MAE |
|---|---:|---:|---:|
| ReLU | 0.955915 | 2.143613 | 1.483034 |
| Sigmoid | 0.828028 | 4.233777 | 3.025304 |
| Tanh | 0.812117 | 4.425301 | 3.115176 |

**Best observed activation function: ReLU**

Graphs generated:

- Activation Function vs RMSE
- Activation Function vs MAE
- Activation Function vs R² Score

### 7.3 Learning Rate Experiment

Four learning rates were tested:

- 0.1
- 0.01
- 0.001
- 0.0001

The architecture was kept at two hidden layers with 64 neurons per layer and ReLU activation.

**Best observed learning rate in this experiment: 0.1**

- R² Score: 0.995917
- RMSE: 0.652358
- MAE: 0.533181

Graphs generated:

- Learning Rate vs RMSE
- Learning Rate vs MAE
- Learning Rate vs R² Score

**Note:** The best parameter from an individual experiment does not necessarily produce the best combined ANN configuration. A high learning rate can behave differently when used with a deeper network.

---

## 8. Final ANN Model

The individually best parameter values were initially combined and tested. However, the combination of five hidden layers and a learning rate of 0.1 produced unstable and poor test performance.

Therefore, a more stable configuration was trained using:

| Parameter | Final Configuration |
|---|---|
| Hidden layers | 5 |
| Neurons per hidden layer | 64 |
| Activation function | ReLU |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Batch size | 32 |
| Maximum epochs | 200 |
| Early stopping patience | 15 |

Early stopping was used to monitor validation loss and restore the best-performing model weights.

### Final ANN Performance

| Evaluation Metric | Value |
|---|---:|
| R² Score | 0.994349 |
| RMSE | 0.767504 |
| MAE | 0.604282 |

The final ANN model achieved a high R² score and low prediction errors on the test dataset.

Graphs generated:

- ANN Actual vs Predicted
- ANN Training vs Validation Loss

---

## 9. Model Evaluation Metrics

Three evaluation metrics were used to compare the models.

### R² Score – Coefficient of Determination

R² indicates how well the model explains the variation in the target variable.

A value closer to 1 indicates a better fit.

### RMSE – Root Mean Squared Error

RMSE measures the square root of the average squared difference between actual and predicted values.

A lower RMSE indicates smaller prediction errors and penalizes larger errors more strongly.

### MAE – Mean Absolute Error

MAE calculates the average absolute difference between actual and predicted values.

A lower MAE indicates better prediction accuracy.

---

## 10. Comparative Analysis: MLR vs KNNR vs ANN

The three models were compared using their Heating Load prediction results.

### Final Comparison Results

| Model | R² Score | RMSE | MAE |
|---|---:|---:|---:|
| MLR | 0.909259 | 3.075400 | 2.263105 |
| KNNR | 0.952007 | 2.236610 | 1.424905 |
| ANN | **0.994349** | **0.767504** | **0.604282** |

### Interpretation

**Multiple Linear Regression (MLR):**

MLR achieved an R² score of 0.909259. It provides a useful baseline for understanding the relationship between building design features and Heating Load.

**K-Nearest Neighbors Regression (KNNR):**

KNNR performed better than MLR, achieving an R² score of 0.952007 and lower prediction errors.

**Artificial Neural Network (ANN):**

ANN achieved the highest R² score of 0.994349 and the lowest RMSE and MAE values.

This indicates that ANN captured the relationships in the dataset more effectively than the other two models in this experiment.

### Comparative Graphs

The following graphs were generated:

- Model Comparison – R² Score
- Model Comparison – RMSE
- Model Comparison – MAE

---

## 11. Key Findings

1. ANN achieved the best predictive performance among the three evaluated models.
2. KNNR performed better than MLR in the Heating Load prediction experiment.
3. The number of hidden layers influenced ANN performance.
4. ReLU performed better than Sigmoid and Tanh in the activation function experiment.
5. The learning rate influenced training behavior and prediction accuracy.
6. Combining individually optimal parameters did not guarantee the best final model.
7. The stable ANN configuration achieved an R² score of 0.994349.

---

## 12. Conclusion

This project demonstrates the application of machine learning and artificial neural networks for predicting building Heating Load using building design parameters.

Three regression models—MLR, KNNR, and ANN—were developed and evaluated using R² Score, RMSE, and MAE.

Among the evaluated models, ANN achieved the best results, with an R² score of 0.994349, RMSE of 0.767504, and MAE of 0.604282.

The experiments also demonstrate that ANN performance depends on architectural and training parameters such as the number of hidden layers, activation functions, and learning rates.

Overall, the study highlights the potential of ANN-based regression for building energy prediction.

---

## 13. Project Folder Structure

```text
Building_Energy_Efficiency_Project/
│
├── Building_Energy_Efficiency.ipynb
│
├── Building Energy Efficiency.csv
│
├── comparison_results.csv
│
├── ANN_Experiments/
│   │
│   ├── ANN_Actual_vs_Predicted.png
│   ├── ANN_Training_Validation_Loss.png
│   │
│   ├── hidden_layers_vs_rmse.png
│   ├── hidden_layers_vs_mae.png
│   ├── hidden_layers_vs_r2.png
│   │
│   ├── activation_vs_rmse.png
│   ├── activation_vs_mae.png
│   ├── activation_vs_r2.png
│   │
│   ├── learning_rate_vs_rmse.png
│   ├── learning_rate_vs_mae.png
│   ├── learning_rate_vs_r2.png
│   │
│   ├── Best_Parameter_Results.csv
│   ├── Best_Parameter_RMSE_Comparison.png
│   ├── Best_Parameter_MAE_Comparison.png
│   ├── Best_Parameter_R2_Comparison.png
│   │
│   ├── All_ANN_Experiments.csv
│   ├── Stable_ANN_Final_Results.csv
│   ├── Final_Model_Comparison.csv
│   ├── MLR_KNNR_Original_Results.csv
│   │
│   ├── Model_Comparison_R2_Score.png
│   ├── Model_Comparison_RMSE.png
│   └── Model_Comparison_MAE.png
│
└── Building_Energy_ANN_Results.zip
```

*The folder structure above represents the intended organization of the project files.*

---

## 14. How to Run the Project

1. Download or clone the project repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook.
4. Ensure the dataset is available in the expected location or download it using KaggleHub.
5. Run the notebook cells in sequential order.
6. Review the model results, experimental tables, and generated graphs.

### Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow kagglehub
```

---

## 15. Limitations and Future Scope

### Limitations

- The analysis uses a single building energy efficiency dataset.
- The ANN experiments use a limited set of architectures and parameter values.
- The final comparison focuses on Heating Load; the ANN has not been evaluated for Cooling Load in this comparison.
- Results are based on a particular training and testing split.
- High accuracy on this dataset does not guarantee the same performance on other building datasets or real-world buildings.



