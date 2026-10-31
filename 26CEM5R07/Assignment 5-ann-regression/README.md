# California Housing Price Prediction Using Artificial Neural Networks (ANN)

## 1. Project Overview

This project implements an Artificial Neural Network (ANN) regression model to predict median house values in California using housing-related characteristics.

The project uses the California Housing dataset, which contains geographical, demographic, and housing-related information. The ANN model learns the relationships between these input features to estimate median house values.

A systematic experimental analysis was conducted using different hidden-layer configurations, activation functions, and learning rates to investigate their influence on regression performance.

The ANN model was also compared with previously developed Multiple Linear Regression (MLR) and K-Nearest Neighbors Regression (KNNR) models.

## 2. Project Objectives

- Understand the fundamentals of Artificial Neural Networks.
- Apply ANN to a real-world regression problem.
- Perform data preprocessing and feature scaling.
- Investigate the influence of different activation functions.
- Experiment with different hidden-layer configurations.
- Evaluate the influence of different learning rates.
- Conduct 81 ANN experiments using different parameter combinations.
- Identify the best-performing ANN configuration.
- Compare ANN performance with MLR and KNNR models.

## 3. Dataset Description

**Dataset:** California Housing Dataset

The dataset contains information about housing districts in California.

### Input Features

The dataset includes eight numerical features:

1. Longitude
2. Latitude
3. Housing Median Age
4. Total Rooms
5. Total Bedrooms
6. Population
7. Households
8. Median Income

It also contains one categorical feature:

- Ocean Proximity

### Target Variable

`median_house_value`

The target variable represents the median house value for a particular district in California.

## 4. Technologies and Libraries

The following technologies and libraries were used:

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

### 5.1 Data Loading and Exploration

- Imported the California Housing dataset using Pandas.
- Examined dataset dimensions and data types.
- Checked for missing values.
- Analysed numerical and categorical features.
- Explored relationships between housing characteristics and median house values.

### 5.2 Data Preprocessing

- Separated independent variables (X) and the dependent variable (y).
- Identified numerical and categorical features.
- Handled missing values using appropriate imputation.
- Converted categorical data into numerical representations using OneHotEncoder.
- Divided the dataset into training and testing subsets.

### 5.3 Feature Scaling

StandardScaler was used to standardize numerical input features.

Feature scaling is important in ANN because input features may have different numerical ranges. Standardization helps prevent features with larger magnitudes from dominating the learning process.

The scaler was fitted using training data and then applied to testing data.

### 5.4 ANN Model Development

The ANN regression model was implemented using Scikit-learn's `MLPRegressor`.

The model consists of interconnected neurons organized into input, hidden, and output layers.

- Input layer: Receives housing-related features.
- Hidden layers: Learn complex relationships and patterns.
- Output layer: Produces the predicted median house value.

### 5.5 Activation Function Experimentation

Three activation functions were investigated:

| Activation Function | Description |
|---|---|
| ReLU | Rectified Linear Unit; introduces non-linearity and is commonly used in hidden layers. |
| Logistic | Sigmoid activation function; produces values between 0 and 1. |
| Tanh | Hyperbolic tangent; produces values between -1 and 1. |

The experiments helped evaluate how different activation functions influence the learning behaviour and prediction performance of the ANN.

## 6. ANN Experimental Design

A total of **81 experiments** were conducted to investigate the effect of different ANN configurations.

### Experimental Parameters

| Parameter | Values |
|---|---|
| Hidden layers | 2 to 10 |
| Neurons per hidden layer | 32 |
| Activation functions | ReLU, Logistic, Tanh |
| Learning rates | 0.0001, 0.001, 0.01 |
| Model | MLPRegressor |
| Evaluation metrics | R², MAE, RMSE |

### Number of Experiments

The total number of experiments was calculated as:

9 hidden-layer configurations × 3 activation functions × 3 learning rates = **81 experiments**

Each experiment was evaluated using the same set of regression metrics to identify the best-performing configuration.

## 7. Best ANN Model Configuration

The best-performing individual ANN configuration was:

| Parameter | Best Configuration |
|---|---|
| Hidden layers | 6 |
| Neurons per hidden layer | 32 |
| Activation function | Tanh |
| Learning rate | 0.001 |

### Best Model Performance

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.803074 |
| MAE | $33,923.78 |
| RMSE | $50,799.01 |

The selected model achieved an R² score of approximately 0.8031, indicating that it explains approximately 80.31% of the variation in median house values on the evaluated test dataset.

## 8. Experimental Analysis

### 8.1 Activation Function Analysis

The experimental results showed that Tanh appeared in seven of the ten highest-performing ANN configurations.

This indicates that Tanh was particularly effective for several of the tested California Housing ANN configurations.

ReLU and Logistic were also investigated to understand their performance under different network architectures and learning rates.

### 8.2 Learning Rate Analysis

Three learning rates were evaluated:

- 0.0001
- 0.001
- 0.01

The learning rate of **0.001** appeared most frequently among the top-performing configurations.

This suggests that 0.001 provided an effective learning rate for the tested ANN configurations.

### 8.3 Hidden-Layer Analysis

The number of hidden layers was varied from 2 to 10.

The results demonstrated that increasing the number of hidden layers did not consistently improve prediction performance.

For example:

| Hidden Layers | R² Score |
|---|---:|
| 4 | 0.798927 |
| 9 | 0.794400 |

The four-hidden-layer configuration achieved a higher R² score than the nine-hidden-layer configuration in the compared results.

This demonstrates that a deeper neural network does not necessarily produce better regression performance.

### 8.4 Overall Experimental Findings

The experimental analysis identified the following observations:

- Tanh appeared in seven of the ten top-performing models.
- Learning rate 0.001 was prominent among the top-performing configurations.
- Increasing network depth did not consistently improve performance.
- The best individual configuration was six hidden layers with Tanh activation and a learning rate of 0.001.

## 9. Comparison with MLR and KNN Regression

The ANN model was compared with previously developed Multiple Linear Regression and K-Nearest Neighbors Regression models.

### Model Performance Comparison

| Model | R² Score | MAE | RMSE |
|---|---:|---:|---:|
| Multiple Linear Regression (MLR) | 0.613866 | 51,810.48 | 71,133.17 |
| KNN Regression (K=7) | 0.709497 | 41,661.51 | 61,699.10 |
| Initial ANN Model | 0.770674 | 37,653.92 | 54,818.84 |
| Best ANN Model | 0.803074 | 33,923.78 | 50,799.01 |

### Comparative Analysis

**Multiple Linear Regression:**

The MLR model achieved an R² score of 0.613866, providing a baseline for comparison.

**K-Nearest Neighbors Regression:**

The KNNR model with K=7 improved performance over MLR, achieving an R² score of 0.709497.

**Initial ANN Model:**

The initial ANN model achieved an R² score of 0.770674, demonstrating improved predictive performance compared with the MLR and KNNR models.

**Best ANN Model:**

The optimized ANN configuration achieved the highest R² score of 0.803074 and the lowest MAE and RMSE among the recorded models.

The results indicate that the best ANN configuration outperformed the recorded MLR and KNNR configurations across all three evaluation metrics.

**Note:** The comparison is based on the recorded results from the previously developed models. It should be interpreted cautiously unless identical train-test splits and preprocessing procedures are verified for every model.

## 10. Evaluation Metrics

The following regression evaluation metrics were used:

### R² Score (Coefficient of Determination)

Measures the proportion of variation in the target variable explained by the regression model.

A higher R² score generally indicates better predictive performance.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted house values.

A lower MAE indicates smaller average prediction errors.

### Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction errors.

RMSE gives greater importance to larger prediction errors.

Together, these metrics provide a comprehensive evaluation of ANN regression performance.

## 11. Project Structure

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
│   ├── ANN_Top_10_Models.csv
│   └── ANN_Best_Model_Results.csv
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
│   └── Model_Comparison_R2.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 12. How to Run the Project

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

Open `california_ann.ipynb` from the `notebook` directory and execute the cells sequentially.

## 13. Limitations and Future Scope

### Limitations

- ANN performance depends on the selected architecture and hyperparameters.
- Training neural networks can require considerable computational resources.
- Prediction errors may be higher for certain housing observations.
- The experimental results depend on the dataset and evaluation procedure.
- Selecting the best configuration based on test-set performance can introduce selection bias.

### Future Scope

- Apply cross-validation for more robust model evaluation.
- Use a separate validation dataset for hyperparameter selection.
- Explore additional ANN architectures.
- Investigate alternative optimization algorithms.
- Compare ANN with other advanced regression algorithms.
- Explore methods to improve prediction accuracy.

## 14. Conclusion

This project demonstrates the application of Artificial Neural Networks for predicting median house values using the California Housing dataset.

A total of 81 experiments were conducted by varying hidden-layer configurations, activation functions, and learning rates.

The best-performing configuration consisted of six hidden layers with 32 neurons per layer, Tanh activation, and a learning rate of 0.001.

The model achieved an R² score of 0.803074, MAE of $33,923.78, and RMSE of $50,799.01.

Comparative analysis showed that the best ANN model achieved better recorded performance than the previously developed MLR and KNNR models.

The project provided practical experience in ANN architecture design, regression modelling, activation function experimentation, hyperparameter analysis, model evaluation, and comparative performance analysis.

## Author

**Vyshnavi Perumal**

Artificial Intelligence and Machine Learning | Geoinformatics
