# Student Performance Prediction Using ANN, Multiple Linear Regression and K-Nearest Neighbours Regression

## Abstract

This project focuses on predicting student academic performance using machine learning regression techniques. The Student Performance dataset is used to develop and evaluate three models: Artificial Neural Network Regression (ANN), Multiple Linear Regression (MLR) and K-Nearest Neighbours Regression (KNNR).

The project is organised into two parts. Part I presents the development and evaluation of the ANN Regression model. A series of experiments is conducted by varying the number of hidden layers, activation functions and learning rates to identify the best-performing ANN configuration. Part II compares the selected ANN model with MLR and KNNR using standard regression evaluation metrics.

The experimental results indicate that MLR achieved the best overall predictive performance, followed closely by ANN Regression, while KNNR produced comparatively higher prediction errors. The project demonstrates how different regression techniques can be applied to predict student performance and how their predictive capabilities can be evaluated and compared.

---

# Part I – Artificial Neural Network Regression

## 1. Introduction

Artificial Neural Networks (ANNs) are machine learning models inspired by the interconnected structure of biological neural networks. They consist of interconnected processing units called neurons, which are organised into layers. During training, the network learns relationships between input variables and the target variable by adjusting the weights and biases associated with its connections.

ANN Regression is used when the objective is to predict a continuous numerical value. In this project, an ANN Regression model is developed to predict the **Performance Index** of students using five input features related to their academic activities and daily habits.

Unlike a simple linear model, an ANN can learn complex and nonlinear relationships between input variables and the target. The performance of an ANN depends on several factors, including the number of hidden layers, number of neurons, activation functions, learning rate and optimisation algorithm.

To investigate these factors, multiple ANN configurations are trained and evaluated. The best-performing configuration is selected based on validation performance and subsequently evaluated using an independent test dataset.

## 2. Dataset Description

### 2.1 Dataset Source

The Student Performance dataset was obtained from Kaggle:

[Student Performance – Multiple Linear Regression Dataset](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)

### 2.2 Dataset Characteristics

The dataset contains 10,000 student records and six columns. Five columns represent the input features, while one column represents the target variable.

| Attribute | Description | Data type |
|---|---|---|
| Hours Studied | Number of hours spent studying | Numerical |
| Previous Scores | Student's previous academic scores | Numerical |
| Extracurricular Activities | Participation in extracurricular activities (Yes/No) | Categorical |
| Sleep Hours | Average number of hours of sleep | Numerical |
| Sample Question Papers Practiced | Number of sample question papers practised | Numerical |
| Performance Index | Student performance value to be predicted | Numerical target |

### 2.3 Dataset Summary

| Dataset characteristic | Value |
|---|---:|
| Total records | 10,000 |
| Total columns | 6 |
| Input features | 5 |
| Numerical input features | 4 |
| Categorical input features | 1 |
| Target variable | Performance Index |
| Missing values | 0 |
| Duplicate records | 127 |

The 127 duplicate records were retained during the experiments.

### 2.4 Target Variable

The target variable is **Performance Index**, which represents student performance as a continuous numerical value.

Since the objective is to estimate a numerical value rather than assign observations to predefined categories, the problem is formulated as a regression task.

## 3. ANN Regression Architecture

### 3.1 General Architecture

The ANN model consists of three main types of layers:

1. **Input layer:** Receives the predictor variables describing the students.
2. **Hidden layers:** Process the input information and learn relationships between the predictor variables and the target.
3. **Output layer:** Produces the predicted Performance Index.

Each neuron receives input values, applies weights and a bias, and passes the resulting value through an activation function.

The network learns by adjusting its weights and biases during training to minimise prediction error.

### 3.2 Hidden Layers and Neurons

Hidden layers are intermediate layers located between the input and output layers. They allow the ANN to learn patterns and relationships that may not be represented by a direct linear relationship.

The number of hidden layers determines the depth of the network, while the number of neurons determines the number of processing units in each hidden layer.

In this experiment, the number of hidden layers was varied from one to seven, with 16 neurons in each hidden layer.

| Parameter | Configurations tested |
|---|---|
| Number of hidden layers | 1, 2, 3, 4, 5, 6 and 7 |
| Neurons per hidden layer | 16 |
| Output layer | One numerical output |

A network with more hidden layers can learn more complex representations, but additional layers do not automatically improve performance. They can also increase model complexity and training requirements.

The purpose of testing different network depths was to identify the architecture that provided the best validation performance for the Student Performance dataset.

### 3.3 Activation Functions

Activation functions determine how the output of a neuron is transformed before being passed to the next layer. They introduce nonlinearity into the network and allow it to learn more complex relationships.

Three activation functions were investigated.

**ReLU (Rectified Linear Unit)**

ReLU returns zero for negative input values and returns the input value itself for positive values.

It is widely used in neural networks because it is computationally efficient and supports effective learning in many applications.

**Sigmoid (Logistic Function)**

The Sigmoid activation function maps input values to an output between zero and one.

It provides a smooth nonlinear transformation. However, for very large positive or negative input values, its gradients can become very small, which may slow down learning.

**Tanh (Hyperbolic Tangent)**

The Tanh activation function maps input values to a range between -1 and 1.

Unlike Sigmoid, its output is centred around zero. It can represent both positive and negative activation values, although it can also experience vanishing gradients for extreme input values.

The three activation functions were tested to determine which one produced the best validation performance.

### 3.4 Learning Rate

The learning rate controls the size of the weight updates made during the training process. It determines how much the network adjusts its weights in response to the prediction error.

Three learning rates were investigated:

| Learning rate | Explanation |
|---|---|
| 0.001 | A relatively small learning rate that produces smaller updates and supports gradual learning. |
| 0.005 | An intermediate learning rate that allows moderately larger updates than 0.001. |
| 0.01 | A relatively larger learning rate that can produce faster updates but may make training less stable if the updates are too large. |

A learning rate that is too small can make training slow, whereas a learning rate that is too large can prevent the model from converging effectively.

Testing different learning rates helps identify a suitable balance between training stability and learning speed.

### 3.5 Optimisation Algorithm – Adam

The ANN models were trained using the Adam optimiser.

Adam is an adaptive optimisation algorithm that adjusts parameter updates using estimates of the gradients and squared gradients. It combines ideas from momentum-based optimisation and adaptive learning-rate methods.

It is widely used for neural network training because it can provide efficient and practical optimisation across different model configurations.

Adam was kept consistent across the ANN experiments so that the influence of hidden layers, activation functions and learning rates could be investigated.

## 4. ANN Regression Methodology

### 4.1 Data Preprocessing

Preprocessing was performed before training the ANN model.

**Categorical encoding**

The `Extracurricular Activities` feature contains Yes/No values. It was converted into numerical representation using one-hot encoding so that it could be used by the neural network.

**Feature standardisation**

The numerical input features were standardised using `StandardScaler`.

Standardisation transforms the features to comparable scales. This is important for ANN training because features with substantially different numerical ranges can affect the optimisation process.

The scaler was fitted using the training data and then applied to the validation and test data.

**Target standardisation**

The target variable, `Performance Index`, was also standardised during ANN training.

After prediction, the ANN output was inverse transformed to the original Performance Index scale before calculating the final evaluation metrics.

### 4.2 Training, Validation and Testing

The dataset was divided into training and testing subsets using an 80:20 split with `random_state=42`.

| Dataset subset | Observations | Percentage |
|---|---:|---:|
| Training and validation pool | 8,000 | 80% |
| Test | 2,000 | 20% |
| Total | 10,000 | 100% |

The 8,000 observations in the training and validation pool were further divided using a 15% validation split.

| Subset | Observations |
|---|---:|
| Training | 6,800 |
| Validation | 1,200 |
| Test | 2,000 |
| **Total** | **10,000** |

The training data were used to fit the model. The validation data were used to compare ANN configurations and select the best configuration. The test data were kept separate for the final evaluation.

### 4.3 ANN Configuration Experiments

A total of 63 ANN configurations were investigated.

The experiments were created by combining:

- Seven hidden-layer configurations (1–7).
- Three activation functions (ReLU, Sigmoid and Tanh).
- Three learning rates (0.001, 0.005 and 0.01).

Each hidden layer contained 16 neurons.

| Experimental parameter | Values |
|---|---|
| Hidden layers | 1–7 |
| Neurons per hidden layer | 16 |
| Activation functions | ReLU, Sigmoid, Tanh |
| Learning rates | 0.001, 0.005, 0.01 |
| Optimiser | Adam |
| Maximum iterations | 1,000 |
| Total configurations | 63 |

The experiments were conducted to determine how variations in the network architecture and training parameters affected prediction performance.

### 4.4 Best Configuration Selection

The validation dataset was used to assess the performance of the 63 configurations.

The configuration with the best validation performance was selected for final testing. The test dataset was not used during configuration selection.

This separation helps provide an independent assessment of the selected model's ability to predict unseen observations.

## 5. Evaluation Metrics

Three standard regression evaluation metrics were used to assess ANN performance: RMSE, MAE and R².

### 5.1 Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared difference between actual and predicted values.

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

A lower RMSE indicates better prediction performance. Since the errors are squared, larger prediction errors have a greater influence on the metric.

### 5.2 Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

A lower MAE indicates that the predicted values are closer to the actual values on average.

### 5.3 Coefficient of Determination (R²)

R² measures the proportion of variation in the target variable explained by the regression model.

\[
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
\]

An R² value closer to 1 indicates that the model explains a larger proportion of the variation in the target variable.

All final ANN metrics were calculated after converting the predictions back to the original Performance Index scale.

## 6. ANN Regression Results and Best Performing ANN

### 6.1 Best Performing ANN Configuration

After evaluating all 63 configurations, the best-performing ANN was identified as follows:

| Parameter | Best-performing value |
|---|---|
| Number of hidden layers | 1 |
| Neurons per hidden layer | 16 |
| Activation function | ReLU |
| Learning rate | 0.005 |
| Optimiser | Adam |
| Maximum iterations | 1,000 |

The selected model consisted of one hidden layer containing 16 neurons and used ReLU activation with a learning rate of 0.005.

This configuration achieved the best validation performance among the tested ANN configurations.

### 6.2 ANN Test Performance

The selected ANN model was evaluated on the independent test dataset containing 2,000 observations.

| Evaluation metric | ANN test result |
|---|---:|
| RMSE | 2.0463 |
| MAE | 1.6364 |
| R² | 0.9887 |

The ANN achieved an R² of 0.9887, indicating that it explained approximately 98.87% of the variation in the test target values.

The RMSE and MAE values indicate that the model's predictions were relatively close to the actual Performance Index values.

### 6.3 Interpretation of ANN Results

The results demonstrate that ANN Regression was able to learn the relationship between the student-related predictor variables and Performance Index.

The best-performing configuration used one hidden layer rather than a deeper network. This indicates that increasing the number of hidden layers does not necessarily improve prediction performance for every dataset.

The selected combination of ReLU activation and a learning rate of 0.005 provided the best validation result among the 63 configurations tested.

## 7. ANN Output Tables and Plots – Explanation

This section explains the purpose and interpretation of the ANN experiment outputs.

### 7.1 ANN Configuration Results Table

The ANN configuration results table records the 63 experiments conducted using different combinations of hidden layers, activation functions and learning rates.

Each row represents one configuration. The table allows the validation performance of the tested configurations to be compared and helps identify the best-performing ANN.

Important information includes the number of hidden layers, activation function, learning rate and validation metrics.

### 7.2 ANN Performance Comparison Plot

An ANN performance comparison plot can be used to examine how prediction performance changes with different experimental configurations.

It helps identify whether changing the network depth, activation function or learning rate improves or reduces model performance.

For error metrics such as RMSE and MAE, lower values indicate better performance. For R², higher values indicate better performance.

### 7.3 Actual vs. Predicted Plot

An actual-versus-predicted plot compares the actual Performance Index values with the values predicted by the ANN.

Points close to the 45-degree reference line indicate accurate predictions. Larger deviations from this line indicate larger prediction errors.

This plot provides a visual assessment of the model's prediction accuracy.

### 7.4 Residual Plot

A residual plot displays the differences between actual and predicted values against the predicted values.

\[
Residual=y-\hat{y}
\]

Residuals distributed around zero without a clear systematic pattern generally indicate that the model is not consistently overpredicting or underpredicting across the prediction range.

### 7.5 ANN Performance Metrics Table

The ANN performance metrics table summarises the selected model's final test results using RMSE, MAE and R².

It provides the numerical basis for assessing the ANN and comparing it with the other regression models in Part II.

*The descriptions of plots and tables should be retained in the final report only where the corresponding output was actually generated by the ANN experiment.*

---

# Part II – Comparative Analysis of MLR, KNNR and ANN Regression

## 8. Introduction to Regression Model Comparison

The second part of the project compares three regression techniques for predicting student performance:

1. Multiple Linear Regression (MLR).
2. K-Nearest Neighbours Regression (KNNR).
3. Artificial Neural Network Regression (ANN).

The objective is to evaluate the predictive performance of the three models using the Student Performance dataset and determine which model provides the most accurate predictions.

All three models were assessed using RMSE, MAE and R².

## 9. Overview of the Regression Models

### 9.1 Multiple Linear Regression (MLR)

Multiple Linear Regression is a statistical technique that models the relationship between one continuous dependent variable and multiple independent variables.

It estimates the target variable as a linear combination of the input features.

The general equation is:

\[
Y=\beta_0+\beta_1X_1+\beta_2X_2+\cdots+\beta_nX_n+\epsilon
\]

Where:

- \(Y\) represents the predicted target variable.
- \(\beta_0\) is the intercept.
- \(\beta_1,\beta_2,\ldots,\beta_n\) are the regression coefficients.
- \(X_1,X_2,\ldots,X_n\) are the input variables.
- \(\epsilon\) represents the error term.

In this project, MLR was implemented using the `LinearRegression` algorithm from scikit-learn.

### 9.2 K-Nearest Neighbours Regression (KNNR)

K-Nearest Neighbours Regression is a non-parametric regression technique that predicts the target value of a new observation using the target values of its nearest training observations.

The algorithm calculates distances between the new observation and training observations, identifies the nearest neighbours and estimates the target value using their target values.

The KNNR model in this project used:

| Parameter | Value |
|---|---|
| Number of neighbours (K) | 5 |
| Algorithm | KNeighborsRegressor |
| Feature scaling | StandardScaler |

Feature scaling is important for KNNR because the algorithm uses distance calculations. Without scaling, features with larger numerical ranges may dominate the distance measurement.

### 9.3 Artificial Neural Network Regression (ANN)

ANN Regression uses interconnected neurons arranged in layers to learn relationships between input variables and a continuous target variable.

The selected ANN model used the following configuration:

| Parameter | Value |
|---|---|
| Hidden layers | 1 |
| Neurons per hidden layer | 16 |
| Activation function | ReLU |
| Learning rate | 0.005 |
| Optimiser | Adam |

The ANN model was developed using `MLPRegressor` from scikit-learn.

## 10. Comparative Experimental Methodology

### 10.1 Dataset and Preprocessing

The same Student Performance dataset was used for the three regression models.

The five predictor variables were used to estimate the target variable, Performance Index.

The categorical feature, `Extracurricular Activities`, was converted into numerical representation.

For KNNR, numerical feature scaling was performed using `StandardScaler`, fitted on the training data and then applied to the test data.

For ANN, numerical inputs were standardised, the categorical variable was one-hot encoded, and the target variable was standardised. ANN predictions were inverse transformed before calculating evaluation metrics.

### 10.2 Training and Testing Split

An 80:20 train-test split was used with `random_state=42`.

| Dataset subset | Observations | Percentage |
|---|---:|---:|
| Training | 8,000 | 80% |
| Testing | 2,000 | 20% |
| **Total** | **10,000** | **100%** |

The training dataset was used to fit the models, while the test dataset was used to assess their prediction performance.

### 10.3 Model Training

The three models were trained using their respective algorithms:

- MLR: `LinearRegression`
- KNNR: `KNeighborsRegressor`, K=5
- ANN: selected `MLPRegressor` configuration

Their reported results were compared using RMSE, MAE and R².

### 10.4 Evaluation Criteria

The models were compared using the following criteria:

- Lower RMSE indicates smaller squared prediction errors.
- Lower MAE indicates smaller average absolute prediction errors.
- Higher R² indicates a greater proportion of target variation explained by the model.

## 11. Comparative Results

### 11.1 Overall Model Performance

The following table presents the test performance of MLR, KNNR and ANN Regression.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| MLR | 2.020552 | 1.611121 | 0.988983 |
| KNNR | 2.925932 | 2.360400 | 0.976898 |
| ANN | 2.046300 | 1.636400 | 0.988700 |

### 11.2 Best Performing Regression Model

<ins>Multiple Linear Regression (MLR) was the best-performing regression model in this experiment.</ins>

It achieved:

| Evaluation metric | MLR test result |
|---|---:|
| RMSE | 2.020552 |
| MAE | 1.611121 |
| R² | 0.988983 |

MLR achieved the lowest RMSE and MAE and the highest R² among the three models.

This indicates that MLR provided the most accurate predictions for the Student Performance dataset under the reported experimental setup.

### 11.3 Best Performing ANN Model

The best-performing ANN configuration was:

- One hidden layer.
- 16 neurons.
- ReLU activation.
- Learning rate of 0.005.
- Adam optimiser.

Its test performance was:

| Evaluation metric | ANN test result |
|---|---:|
| RMSE | 2.0463 |
| MAE | 1.6364 |
| R² | 0.9887 |

The ANN achieved performance very close to MLR, although its error values were slightly higher and its R² was marginally lower.

### 11.4 KNNR Performance

KNNR achieved the following results:

| Evaluation metric | KNNR test result |
|---|---:|
| RMSE | 2.925932 |
| MAE | 2.360400 |
| R² | 0.976898 |

KNNR produced higher RMSE and MAE values and a lower R² than MLR and ANN. Therefore, it achieved the lowest predictive performance among the three models in this experiment.

### 11.5 Overall Model Ranking

Based on the three evaluation metrics, the overall ranking is:

| Rank | Model | Reason |
|---:|---|---|
| 1 | MLR | Lowest RMSE and MAE and highest R² |
| 2 | ANN | Performance close to MLR, with slightly higher errors |
| 3 | KNNR | Highest RMSE and MAE and lowest R² |

MLR was the best overall regression model, followed by ANN and KNNR.

## 12. Explanation of Comparative Plots and Tables

### 12.1 RMSE Comparison Plot

The RMSE comparison plot displays the RMSE values obtained by MLR, KNNR and ANN.

A lower value indicates smaller squared prediction errors and therefore better performance according to RMSE.

MLR achieved the lowest RMSE, followed closely by ANN. KNNR recorded the highest RMSE.

### 12.2 MAE Comparison Plot

The MAE comparison plot presents the average absolute prediction error for each model.

A lower MAE indicates that the predictions are closer to the actual Performance Index values on average.

MLR achieved the lowest MAE, while KNNR recorded the highest MAE.

### 12.3 R² Comparison Plot

The R² comparison plot displays the proportion of variation in Performance Index explained by each model.

A higher R² indicates better explanatory performance.

MLR achieved the highest R², followed closely by ANN. KNNR produced the lowest R².

### 12.4 Model Performance Comparison Table

The model performance comparison table provides the numerical results of all three regression models using RMSE, MAE and R².

It allows the predictive performance of the models to be compared using the same evaluation criteria.

### 12.5 Model Performance Ranking Table

The model performance ranking table presents the models in order of their overall predictive performance.

It provides a concise summary of the best-performing model and the relative performance of the other models.

*Retain only plot and table descriptions that correspond to files actually generated by the comparison experiment.*

## 13. Overall Discussion

The comparative results show that all three regression models were able to predict student performance, but their predictive performance differed.

MLR achieved the best overall performance, with an RMSE of 2.020552, MAE of 1.611121 and R² of 0.988983.

ANN Regression produced results close to MLR, with an RMSE of 2.0463, MAE of 1.6364 and R² of 0.9887. This indicates that the ANN was also highly effective in learning the relationship between the predictor variables and Performance Index.

KNNR achieved lower performance than the other two models, as reflected by its higher error values and lower R².

The close performance of MLR and ANN suggests that the relationship between the selected student characteristics and Performance Index was represented effectively by a linear model. In this experiment, the additional flexibility of ANN did not provide a meaningful improvement over MLR.

The results also demonstrate that a more complex machine learning model does not necessarily produce better predictive accuracy. Model selection should be based on measured performance rather than model complexity alone.

## 14. Conclusion

This project developed and evaluated three regression models for predicting student performance using the Student Performance dataset.

In Part I, ANN Regression was developed and evaluated through 63 experimental configurations. The number of hidden layers, activation functions and learning rates were varied to identify the best-performing ANN architecture.

The selected ANN configuration consisted of one hidden layer with 16 neurons, ReLU activation and a learning rate of 0.005. It achieved an RMSE of 2.0463, MAE of 1.6364 and R² of 0.9887 on the independent test dataset.

In Part II, ANN Regression was compared with Multiple Linear Regression and K-Nearest Neighbours Regression.

MLR achieved the best overall performance, with an RMSE of 2.020552, MAE of 1.611121 and R² of 0.988983. ANN ranked second, demonstrating performance close to MLR, while KNNR ranked third.

Therefore, **Multiple Linear Regression was the best-performing regression model for the Student Performance dataset under the reported experimental setup**, while the one-hidden-layer ReLU network was the best-performing ANN configuration.

Overall, the project demonstrates the application of statistical regression, instance-based learning and neural network methods to a student performance prediction problem.

## 15. Software and Libraries Used

The experiments were implemented in Python using the following libraries:

| Library | Purpose |
|---|---|
| Pandas | Dataset loading, processing and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Regression models, preprocessing, data splitting and evaluation |
| Matplotlib | Performance visualisation and plotting |
| Seaborn | Statistical visualisation, where applicable |

The following scikit-learn algorithms were used:

- `LinearRegression`
- `KNeighborsRegressor`
- `MLPRegressor`

## 16. Reproducibility Notes

The experiments used the following settings:

| Parameter | Setting |
|---|---|
| Dataset | Student_Performance.csv |
| Total observations | 10,000 |
| Train-test split | 80:20 |
| Random state | 42 |
| MLR algorithm | LinearRegression |
| KNNR algorithm | KNeighborsRegressor |
| KNNR neighbours | 5 |
| ANN algorithm | MLPRegressor |
| ANN hidden layers tested | 1–7 |
| Neurons per hidden layer | 16 |
| ANN activation functions | ReLU, Sigmoid, Tanh |
| ANN learning rates | 0.001, 0.005, 0.01 |
| ANN optimiser | Adam |
| ANN maximum iterations | 1,000 |

The comparison uses the reported test metrics from the three regression experiments. The ANN experiment used a separate validation subset for selecting its configuration.

For a strictly controlled comparison, all models should be evaluated using identical test observations and consistent preprocessing procedures. The reported train-test split settings are documented above so that the experiments can be reproduced and checked.