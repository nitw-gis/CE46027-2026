# California Housing Price Prediction Using Artificial Neural Networks (ANN)

## Project Overview

This project develops an Artificial Neural Network (ANN) regression model to predict median house values in California using housing-related features.

The model learns the relationships between geographical, demographic, and housing characteristics to estimate house prices. The project uses Python and Scikit-learn to implement, train, and evaluate the ANN model.

## Objectives

- Understand the working principles of Artificial Neural Networks.
- Prepare and preprocess the California Housing dataset.
- Apply feature scaling and categorical feature encoding.
- Implement an ANN regression model using Scikit-learn.
- Explore different activation functions.
- Evaluate model performance using regression metrics.
- Understand the effect of ANN architecture and activation functions on prediction performance.

## Dataset

**Dataset:** California Housing Dataset

The dataset contains information about housing districts in California.

### Input Features

The dataset includes numerical features such as:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income

It also contains the categorical feature:

- Ocean Proximity

### Target Variable

`median_house_value`

The target variable represents the median house value for a particular district.

## Technologies and Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- GitHub

## Methodology

The following workflow was followed:

1. **Data Loading:** Imported the California Housing dataset using Pandas.
2. **Data Exploration:** Examined dataset dimensions, data types, missing values, and feature distributions.
3. **Data Preprocessing:** Separated input features and the target variable.
4. **Categorical Encoding:** Converted the categorical Ocean Proximity feature into a numerical representation.
5. **Train-Test Split:** Divided the dataset into training and testing subsets.
6. **Feature Scaling:** Standardized numerical features to improve ANN training.
7. **ANN Model Development:** Implemented an ANN regression model using Scikit-learn's `MLPRegressor`.
8. **Activation Functions:** Explored ReLU, Logistic (Sigmoid), and Tanh activation functions.
9. **Model Training:** Trained the neural network using the prepared training data.
10. **Prediction and Evaluation:** Generated predictions and evaluated model performance.

## ANN Model Performance

The trained ANN model achieved the following performance:

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.7707 |
| Mean Absolute Error (MAE) | 37,653.92 |
| Root Mean Squared Error (RMSE) | 54,818.84 |

### Performance Interpretation

- **R² Score:** The model explains approximately 77.07% of the variation in the target variable on the test dataset.
- **MAE:** The average absolute prediction error is approximately $37,653.92.
- **RMSE:** The root mean squared prediction error is approximately $54,818.84.

These metrics indicate that the ANN model learned meaningful relationships between housing characteristics and median house values.

## Project Structure

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

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/california-housing-ann.git
```

### 2. Navigate to the Project Directory

```bash
cd california-housing-ann
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook inside the `notebook` directory and execute the cells sequentially.

## Learning Outcomes

Through this project, the following concepts were explored:

- Fundamentals of Artificial Neural Networks.
- Neurons, hidden layers, weights, and biases.
- Activation functions and their importance.
- Feature standardization.
- ANN regression using Scikit-learn.
- Actual versus predicted value comparison.
- Regression evaluation metrics: R², MAE, and RMSE.

## Conclusion

The project demonstrates the application of Artificial Neural Networks to a real-world housing price prediction problem.

The trained ANN achieved an R² score of 0.7707, indicating that it captured a substantial proportion of the variation in California housing values.

This project provided practical experience in neural network development, data preprocessing, model training, prediction, and performance evaluation.

## Author

**Vyshnavi Perumal**

*Artificial Intelligence and Machine Learning | Geoinformatics*
