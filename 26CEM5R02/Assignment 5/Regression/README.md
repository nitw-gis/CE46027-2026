# Assignment 5: Artificial Neural Network (ANN) Regression for Insurance Expense Prediction

## 1. Objective

The objective of this assignment is to develop and evaluate an
Artificial Neural Network (ANN) regression model using the Insurance
Premium Prediction dataset.

The model predicts individual insurance expenses from demographic and
insurance-related input variables. The evaluation considers regression
accuracy, computational resource utilization, execution time, and the
effect of ANN hyperparameters on model performance.

The ANN experiments examine: - The number of hidden layers. - The
activation function. - The learning rate. - Prediction performance using
R², RMSE and MAE. - Computational performance using wall-clock time, CPU
process time and memory change.

## 2. Dataset

### Dataset Name

Insurance Premium Prediction

### Dataset Source

Kaggle

### Dataset URL

https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction/versions/1

### Dataset Description

The dataset contains information related to individuals and their
insurance expenses. It contains 1,338 observations and 7 columns.

The predictor variables used are:

-   `age`
-   `sex`
-   `bmi`
-   `children`
-   `smoker`
-   `region`

The target variable is:

-   `expenses`

### Feature Types

**Numerical features** - `age` - `bmi` - `children`

**Categorical features** - `sex` - `smoker` - `region`

## 3. Software and Tools Used

The implementation uses:

-   Python
-   Anaconda
-   Spyder or Jupyter Notebook
-   pandas
-   NumPy
-   scikit-learn
-   psutil
-   matplotlib
-   Python `time` module

## 4. Data Preprocessing

The dataset is loaded using pandas and inspected for its dimensions,
column names, missing values and data types. The predictor variables are
separated from the target variable (`expenses`).

Numerical and categorical features are processed separately.

### Numerical Features

-   `age`
-   `bmi`
-   `children`

Numerical input features are standardized using `StandardScaler`.

### Categorical Features

-   `sex`
-   `smoker`
-   `region`

Categorical variables are converted into numerical representations using
One-Hot Encoding with `drop="first"` and `handle_unknown="ignore"`.

The target variable (`expenses`) is also standardized during ANN
training. Predictions are converted back to the original expense scale
before calculating R², RMSE and MAE.

## 5. Train, Validation and Test Split

The updated ANN workflow uses three data subsets:

-   Training data: approximately 60%
-   Validation data: approximately 20%
-   Final testing data: 20%

The initial test split uses `test_size=0.20` and `random_state=42`. The
remaining 80% is divided again, with 25% of that portion used for
validation. This gives an approximate 60:20:20 split.

All ANN configurations are compared using the validation set. After
selecting the configuration with the lowest validation RMSE, the
selected model is retrained using the combined training and validation
data and evaluated on the untouched test set.

## 6. Artificial Neural Network Regression

The ANN model is implemented using the `MLPRegressor` class from
scikit-learn.

A Multi-Layer Perceptron consists of an input layer, one or more hidden
layers and an output layer. During training, the network adjusts its
weights and biases to learn the relationship between the input variables
and insurance expenses.

### 6.1 Hidden-Layer Configurations

The number of hidden layers is varied from 1 to 7. Each hidden layer
contains 64 neurons.

The architectures are:

-   1 hidden layer: `(64,)`
-   2 hidden layers: `(64, 64)`
-   3 hidden layers: `(64, 64, 64)`
-   4 hidden layers: four layers with 64 neurons each
-   5 hidden layers: five layers with 64 neurons each
-   6 hidden layers: six layers with 64 neurons each
-   7 hidden layers: seven layers with 64 neurons each

### 6.2 Activation Functions

Three activation functions are evaluated:

-   **ReLU (`relu`)** -- Rectified Linear Unit
-   **Tanh (`tanh`)** -- Hyperbolic tangent
-   **Logistic (`logistic`)** -- Sigmoid activation function

### 6.3 Learning Rates

The following initial learning rates are tested:

-   `0.001`
-   `0.01`
-   `0.1`

The learning rate controls the step size used to update the network
weights during training.

### 6.4 Total ANN Experiments

The experiment combines:

-   7 hidden-layer configurations
-   3 activation functions
-   3 learning rates

Therefore, the total number of ANN configurations is:

**7 × 3 × 3 = 63 configurations**

The model uses the Adam optimizer, a maximum of 500 iterations, early
stopping, a validation fraction of 0.10 and `random_state=42`.

## 7. Evaluation Metrics

The ANN regression model is evaluated using the following metrics.

### R² Score

R² measures the proportion of variation in the target variable explained
by the regression model. A value closer to 1 generally indicates that
the model explains a greater proportion of the variation in insurance
expenses.

### Root Mean Squared Error (RMSE)

RMSE is the square root of the average squared difference between actual
and predicted expenses. Squaring the errors gives greater weight to
larger prediction errors.

### Mean Absolute Error (MAE)

MAE is the average absolute difference between actual and predicted
expenses. It represents the average magnitude of prediction error in the
original expense units.

RMSE and MAE are calculated after transforming the ANN predictions back
to the original `expenses` scale.

## 8. Computational Performance

In addition to prediction accuracy, computational performance is
recorded for each of the 63 ANN configurations.

The recorded measures are:

-   Wall-clock execution time.
-   CPU process time.
-   Memory change in MB.
-   Number of training iterations.
-   Whether training stopped before reaching the maximum iteration
    count.

Wall-clock time is measured using `time.perf_counter()`. CPU process
time is measured using `time.process_time()`. Process memory is obtained
using `psutil`.

Memory change is the difference between the process memory measurements
taken before and after the measured training operation. It is not a
measurement of peak memory consumption. CPU process time can differ from
wall-clock time, including being greater when multiple CPU threads are
used.

## 9. ANN Parameter Analysis and Plots

The updated program generates **13 plot images**.

### Hidden-Layer Analysis: 7 Plots

One plot is generated for each hidden-layer count from 1 to 7. Each plot
contains three metric panels:

-   RMSE
-   MAE
-   R²

Within each metric panel, activation functions and learning rates are
compared.

### Activation-Function Analysis: 3 Plots

One plot is generated for each activation function:

-   ReLU
-   Tanh
-   Logistic

Each plot contains RMSE, MAE and R² panels across the seven hidden-layer
configurations, with separate lines for the three learning rates.

### Model Evaluation Plots: 3 Plots

The remaining plots are:

1.  Best model training-loss curve.
2.  Actual versus predicted insurance expenses.
3.  Residual error analysis, containing residuals versus predicted
    expenses and the residual-error distribution.

The 13 images are saved in the `ANN_Regression_13_Plots` folder.

## 10. Results

The program selects the ANN configuration with the lowest validation
RMSE. The selected configuration is then refitted on the combined
training and validation data and evaluated on the held-out test set.

The final test evaluation reports:

-   R²
-   RMSE
-   MAE
-   Final training wall-clock time
-   Final training CPU time
-   Final training memory change
-   Test prediction time

The numerical ANN results are generated when the Python script is
executed. They are not fixed in this README because they depend on the
actual run and computational environment.

## 11. Code Workflow

The Python program follows these steps:

1.  Load `insurance.csv`.
2.  Inspect dataset dimensions, missing values and data types.
3.  Separate predictor variables and the target variable.
4.  Identify numerical and categorical features.
5.  Split the data into training, validation and test subsets.
6.  Standardize numerical input features.
7.  One-hot encode categorical input features.
8.  Standardize the target variable for ANN training.
9.  Configure hidden-layer counts from 1 to 7.
10. Configure ReLU, Tanh and Logistic activation functions.
11. Test learning rates 0.001, 0.01 and 0.1.
12. Train and validate all 63 ANN configurations.
13. Calculate validation R², RMSE and MAE.
14. Measure wall-clock time, CPU process time and memory change.
15. Rank configurations using validation RMSE.
16. Refit the selected configuration using training and validation data.
17. Evaluate the selected model on the untouched test data.
18. Save experiment metrics and best-model predictions as CSV files.
19. Generate the 13 required plots.
20. Display the selected configuration and final test metrics in the
    console.

## 12. Output Files

The program automatically creates the folder:

`ANN_Regression_13_Plots`

The principal output files are:

### Experiment Results

-   `ann_all_63_experiments.csv` -- results and parameter settings for
    all 63 ANN configurations.
-   `ann_ranked_63_experiments.csv` -- configurations ranked by
    validation RMSE.
-   `ann_best_configuration_metrics.csv` -- selected configuration and
    its validation and held-out test metrics.
-   `ann_best_actual_vs_predicted.csv` -- actual expenses, predicted
    expenses and residuals for the final test set.
-   `ann_resource_comparison.csv` -- computational measurements for all
    configurations.

### Plot Files

**Hidden-layer plots** - `01_hidden_layer_1.png` -
`02_hidden_layer_2.png` - `03_hidden_layer_3.png` -
`04_hidden_layer_4.png` - `05_hidden_layer_5.png` -
`06_hidden_layer_6.png` - `07_hidden_layer_7.png`

**Activation-function plots** - `08_activation_relu.png` -
`09_activation_tanh.png` - `10_activation_logistic.png`

**Model evaluation plots** - `11_best_model_training_loss.png` -
`12_actual_vs_predicted.png` - `13_residual_error_analysis.png`

## 13. How to Run the Program

### Step 1

Install Anaconda.

### Step 2

Activate the Python environment used for the assignment.

### Step 3

Install the required libraries if they are not already available:

``` bash
pip install pandas numpy scikit-learn psutil matplotlib
```

### Step 4

Place `insurance.csv` in the same directory as
`ANN_Insurance_Regression_13_Plots.py`.

### Step 5

Run the Python program in Spyder or from Anaconda Prompt:

``` bash
python ANN_Insurance_Regression_13_Plots.py
```

### Step 6

The selected ANN configuration and final test metrics are displayed in
the console. The CSV files and 13 plot images are saved in
`ANN_Regression_13_Plots`.

## 14. Reproducibility and Limitations

-   The random state is fixed at 42 for the data splitting and ANN model
    initialization.
-   The same dataset, preprocessing steps and parameter settings are
    required for comparable results.
-   Execution time, CPU time and memory measurements may vary with
    hardware, software versions and system load.
-   The memory-change measurement is a before-and-after process memory
    difference, not peak memory usage.
-   The experiment evaluates the selected range of hidden layers,
    activation functions and learning rates; it is not an exhaustive
    search of every possible ANN architecture or hyperparameter.
-   Hyperparameter selection uses validation RMSE, while the final
    reported test metrics are calculated on a separate held-out test
    set.

## 15. Conclusion

This assignment implements an ANN regression workflow for predicting
insurance expenses. It evaluates 63 combinations of hidden-layer
configurations, activation functions and learning rates. Model
performance is examined using R², RMSE and MAE, while computational
performance is assessed using execution time, CPU process time and
memory change.

The final set of 13 plots presents the hidden-layer analysis,
activation-function analysis, training behavior, prediction agreement
and residual errors without generating a large number of repetitive
figures.

## 16. Author

**AI/ML Regression Assignment**

Dataset: Insurance Premium Prediction\
Implementation: Python / Anaconda / Spyder or Jupyter Notebook\
Machine Learning Library: scikit-learn
