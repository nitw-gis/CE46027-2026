# Crop Yield Prediction Using MLR, KNNR and ANN Regression

## 1. Project Overview

This project focuses on predicting crop yield using three machine learning regression techniques:

1. Multiple Linear Regression (MLR)
2. K-Nearest Neighbors Regression (KNNR)
3. Artificial Neural Network Regression (ANNR)

The main objective is to compare the performance of these models and identify the most suitable model for crop yield prediction.

## 2. Dataset Description

**Dataset:** `dataset.csv`

**Target Variable:**
- `Crop_Yield_ton_per_hectare`

**Input Variables:**
- Agricultural and environmental features available in the dataset.
- Numerical and categorical variables.

Categorical variables are converted into numerical format using one-hot encoding.

## 3. Methodology

### Step 1: Data Preprocessing
- Load the dataset using Pandas.
- Remove unnecessary spaces from column names.
- Separate input features and target variable.
- Apply one-hot encoding to categorical features.
- Split the dataset into 80% training data and 20% testing data.
- Apply StandardScaler for feature normalization.

### Step 2: Multiple Linear Regression (MLR)

MLR is implemented as a baseline regression model to predict crop yield using multiple independent variables.

### Step 3: K-Nearest Neighbors Regression (KNNR)

KNNR predicts crop yield based on the nearest observations in the training dataset.

- Number of neighbours (K): 5

### Step 4: Artificial Neural Network Regression (ANNR)

An ANN model is developed using two hidden layers.

The following parameters are tested:

| Parameter | Values |
|---|---|
| Activation Functions | ReLU, Tanh, ELU |
| Learning Rates | 0.1, 0.01, 0.001 |
| Hidden Layer Architectures | (128, 64), (64, 32) |
| Maximum Epochs | 100 |
| Batch Size | 32 |
| Early Stopping Patience | 10 |

A total of 18 ANN experiments are performed:

3 activation functions × 3 learning rates × 2 architectures = **18 experiments**.

The best ANN configuration is selected based on the lowest validation RMSE.

## 4. Performance Evaluation Metrics

The models are evaluated using the following regression metrics:

- **R² Score:** Measures how well the model explains the variation in crop yield. Higher values indicate better performance.
- **RMSE:** Measures the square root of the average squared prediction error. Lower values indicate better performance.
- **MAE:** Measures the average absolute prediction error. Lower values indicate better performance.

## 5. Experimental Results

### 5.1 Final Model Comparison

The following results were obtained from the test dataset:

| Model | R² Score | RMSE | MAE |
|---|---:|---:|---:|
| MLR | **0.9980** | **1.0810** | **0.8638** |
| KNNR | 0.9445 | 5.6896 | 2.9864 |
| ANN | 0.9975 | 1.2039 | 0.9668 |

### 5.2 Best ANN Configuration

Out of the 18 ANN experiments, the following configuration achieved the lowest validation RMSE:

| Parameter | Best Value |
|---|---|
| Activation Function | Tanh |
| Learning Rate | 0.01 |
| Hidden Layer Architecture | (64, 32) |
| Epochs | 35 |
| Validation RMSE | 1.138905 |

### 5.3 Results Interpretation

**Multiple Linear Regression (MLR):**
- Achieved the highest R² score of 0.9980.
- Obtained the lowest RMSE of 1.0810.
- Obtained the lowest MAE of 0.8638.
- Demonstrated the best overall performance among the three models.

**K-Nearest Neighbors Regression (KNNR):**
- Achieved an R² score of 0.9445.
- Recorded an RMSE of 5.6896.
- Showed comparatively lower predictive performance.

**Artificial Neural Network Regression (ANN):**
- Achieved an R² score of 0.9975.
- Recorded an RMSE of 1.2039.
- Demonstrated excellent predictive performance, close to MLR.
- The Tanh activation function with a 0.01 learning rate and (64, 32) architecture was the best configuration.

## 6. Visualizations

Two plots are generated:

**Plot 1: Effect of Activation Functions**
- Compares ReLU, Tanh and ELU.
- Uses average validation RMSE across the tested learning rates and architectures.
- Helps identify the better-performing activation function.

**Plot 2: Model Performance Comparison**
- Compares MLR, KNNR and ANN.
- Displays R² and RMSE values.
- Helps identify the best-performing regression model.

## 7. Final Conclusion

Based on the experimental results, Multiple Linear Regression achieved the best overall performance, with an R² score of 0.9980, RMSE of 1.0810 and MAE of 0.8638.

The Artificial Neural Network also demonstrated excellent predictive performance, achieving an R² score of 0.9975. The best ANN configuration used the Tanh activation function, a learning rate of 0.01 and hidden layers containing 64 and 32 neurons.

KNNR showed comparatively lower performance, with an R² score of 0.9445.

The results indicate that a simpler linear regression model can outperform a more complex neural network when the dataset has a strong linear relationship between its input features and target variable.

Therefore, MLR was the best-performing model for the given dataset, while ANN provided comparable prediction performance.

## 8. Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Spyder IDE

## 9. How to Run the Project

1. Install the required libraries:

   `pip install pandas numpy matplotlib scikit-learn tensorflow`

2. Keep `dataset.csv` and `ANNR_vs_MLR.py` in the same project folder.

3. Open the Python file in Spyder.

4. Run the program.

5. View the model comparison results and generated plots.

## 10. Output Files

- `ANN_Experiments.csv` – Results of all 18 ANN experiments.
- `Model_Comparison.csv` – Final comparison of MLR, KNNR and ANN.
- Activation Function Comparison Plot.
- Model Performance Comparison Plot.

---

**Project:** Crop Yield Prediction Using Machine Learning Regression Models  
**Models:** MLR, KNNR and ANNR  
**Best Performing Model:** Multiple Linear Regression (MLR)
