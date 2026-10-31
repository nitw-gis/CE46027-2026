# California Housing Price Prediction Using Artificial Neural Networks (ANN)

## 1. Project Overview

This project implements an Artificial Neural Network (ANN) regression model to predict median house values in California using housing-related characteristics.

The model learns relationships between geographical, demographic, and housing features to estimate house prices.

The project uses Python and Scikit-learn's `MLPRegressor` to develop, train, and evaluate an ANN regression model.

The primary objective is to understand how Artificial Neural Networks can be applied to real-world regression problems and how different activation functions influence model performance.

## 2. Project Objectives

- Understand the fundamentals of Artificial Neural Networks.
- Explore ANN architecture and hidden layers.
- Perform data preprocessing and feature scaling.
- Apply categorical encoding.
- Implement ANN regression using Scikit-learn.
- Experiment with different activation functions.
- Evaluate model performance using regression metrics.
- Compare actual and predicted house values.

## 3. Dataset Description

**Dataset:** California Housing Dataset

The dataset contains information about housing districts in California.

### Input Features

The dataset includes the following numerical features:

1. Longitude
2. Latitude
3. Housing Median Age
4. Total Rooms
5. Total Bedrooms
6. Population
7. Households
8. Median Income

It also contains the categorical feature:

- Ocean Proximity

### Target Variable

`median_house_value`

The target represents the median house value for a particular California district.

## 4. Technologies and Libraries

The following tools and libraries were used:

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- GitHub

## 5. Methodology

The project follows a systematic machine-learning workflow.

### 5.1 Data Loading

- Imported the California Housing dataset using Pandas.
- Examined the dataset structure and dimensions.
- Checked data types and missing values.

### 5.2 Exploratory Data Analysis

- Analysed numerical feature distributions.
- Examined relationships between input features and house values.
- Studied the categorical Ocean Proximity feature.

### 5.3 Data Preprocessing

- Separated independent variables (X) and the dependent variable (y).
- Identified numerical and categorical features.
- Encoded categorical data into numerical representations.
- Divided the dataset into training and testing subsets.

### 5.4 Feature Scaling

StandardScaler was applied to standardize numerical input features.

Feature scaling is important in ANN because neural networks use optimization algorithms that can be affected by differences in feature magnitudes.

The scaler was fitted on the training data and then applied to the testing data.

### 5.5 ANN Model Development

The ANN regression model was developed using Scikit-learn's `MLPRegressor`.

The model learns relationships between housing features and median house values through interconnected neurons and hidden layers.

### 5.6 Activation Function Experimentation

Three activation functions were selected and investigated:

| Activation Function | Description |
|---|---|
| ReLU | Rectified Linear Unit; introduces non-linearity and is commonly used in hidden layers. |
| Logistic | Sigmoid activation function; produces values between 0 and 1. |
| Tanh | Hyperbolic tangent; produces values between -1 and 1. |

The experiments helped understand how activation functions influence the learning behaviour of the ANN model.

### 5.7 Model Training and Prediction

- Trained the ANN using the prepared training dataset.
- Generated predictions using the testing dataset.
- Compared actual and predicted house values.
- Evaluated the model using regression performance metrics.

## 6. ANN Model Performance

The trained ANN regression model achieved the following results:

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.7707 |
| Mean Absolute Error (MAE) | 37,653.92 |
| Root Mean Squared Error (RMSE) | 54,818.84 |

### Performance Interpretation

**R² Score: 0.7707**

The model explains approximately 77.07% of the variation in the target variable on the test dataset.

**Mean Absolute Error: 37,653.92**

The average absolute difference between actual and predicted house values is approximately $37,653.92.

**Root Mean Squared Error: 54,818.84**

RMSE measures prediction error while giving greater weight to larger errors.

These results indicate that the ANN model learned meaningful relationships between housing characteristics and median house values.

## 7. Evaluation Metrics

The following regression metrics were used:

- **R² Score:** Measures how much variation in the target variable is explained by the model.
- **MAE:** Measures the average absolute difference between actual and predicted values.
- **RMSE:** Measures the square root of the average squared prediction errors.

These metrics provide different perspectives on regression model performance.

## 8. Project Structure

```text
california-housing-ann/
│
├── data/
│   └── housing.csv
│
├── notebook/
│   └── california_housing_ann.ipynb
│
├── results/
│
├── figures/
│
├── README.md
│
└── requirements.txt
```

## 9. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/california-housing-ann.git
```

### Step 2: Navigate to the Project Directory

```bash
cd california-housing-ann
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Open Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook inside the `notebook` directory and execute the cells sequentially.

## 10. Learning Outcomes

Through this project, the following concepts were explored:

- Fundamentals of Artificial Neural Networks.
- Neurons, weights, biases, and hidden layers.
- Activation functions and their importance.
- Numerical feature standardization.
- Categorical feature encoding.
- ANN regression using Scikit-learn.
- Training and testing datasets.
- Actual versus predicted value comparison.
- Regression evaluation metrics: R², MAE, and RMSE.

## 11. Limitations and Future Scope

### Limitations

- ANN performance depends on model architecture and hyperparameter selection.
- Neural networks may require additional tuning to improve prediction accuracy.
- Large prediction errors can occur for certain housing observations.
- The model's performance depends on the quality and representativeness of the dataset.

### Future Scope

- Experiment with additional hidden-layer configurations.
- Explore different learning rates and optimizers.
- Perform systematic hyperparameter optimization.
- Compare ANN with other regression algorithms.
- Investigate methods to improve prediction accuracy.

## 12. Conclusion

This project demonstrates the application of Artificial Neural Networks to a real-world housing price prediction problem.

The ANN regression model achieved an R² score of 0.7707, with an MAE of 37,653.92 and RMSE of 54,818.84.

The project provided practical experience in data preprocessing, feature scaling, categorical encoding, neural network development, activation function experimentation, prediction, and model evaluation.

Overall, the project demonstrates how Artificial Neural Networks can be applied to predict continuous numerical values using multiple input features.

## Author

**Vyshnavi Perumal**

Artificial Intelligence and Machine Learning | Geoinformatics
