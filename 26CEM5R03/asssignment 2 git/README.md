# Assignment 2: Performance Comparison of MLR and KNNR

## 1. Introduction

This assignment compares the performance of two supervised machine
learning regression algorithms:

1.  Multiple Linear Regression (MLR)
2.  K-Nearest Neighbors Regression (KNNR)

Both models are applied to the selected Concrete Compressive Strength
dataset. The models are evaluated using prediction accuracy metrics and
computational performance measures.

The assignment is implemented in Python using Anaconda and Spyder.

## 2. Dataset

**Dataset:** Concrete Compressive Strength Dataset

**Target variable:** `strength`

All columns except `strength` are used as predictor variables.

## 3. Software and Libraries

-   Anaconda
-   Spyder
-   Python 3
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   psutil
-   openpyxl

## 4. Data Preprocessing

The Excel dataset is loaded using Pandas. The program checks for missing
values and duplicate rows. Duplicate rows, if present, are removed.

The target variable is separated from the predictor variables:

``` python
X = df.drop(columns=["strength"])
y = df["strength"]
```

The dataset is divided into 80% training data and 20% testing data using
`train_test_split` with `random_state=42`.

## 5. Multiple Linear Regression

Multiple Linear Regression is used to predict concrete compressive
strength from multiple input variables.

The model is created using:

``` python
mlr = LinearRegression()
```

It is trained using the training dataset and then used to predict the
test dataset.

## 6. K-Nearest Neighbors Regression

K-Nearest Neighbors Regression is used with `k = 5`.

Because KNN is a distance-based algorithm, `StandardScaler` is applied
before KNN:

``` python
knnr = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])
```

## 7. Accuracy Metrics

The following metrics are used to compare prediction performance.

### R²

R² measures the proportion of variation in the target variable explained
by the model.

Generally, a higher R² indicates better explanatory performance.

### RMSE

Root Mean Squared Error measures the magnitude of prediction errors and
gives greater weight to larger errors.

Generally, a lower RMSE indicates smaller prediction errors.

### MAE

Mean Absolute Error measures the average absolute difference between
actual and predicted values.

Generally, a lower MAE indicates smaller average prediction errors.

## 8. Computational Performance

The assignment also requires computational resource utilization and
execution time.

The program records:

-   Training time
-   Prediction time
-   Total execution time
-   CPU time
-   Peak Python memory usage

Training and prediction times are measured using `time.perf_counter()`.
Peak Python memory is measured using `tracemalloc`, and CPU information
is obtained using `psutil`.

Execution time and memory measurements can vary slightly between
computers.

## 9. Model Comparison

The final comparison table contains:

  Parameter              MLR              KNNR
  ---------------------- ---------------- ----------------
  R²                     Program output   Program output
  RMSE                   Program output   Program output
  MAE                    Program output   Program output
  Training Time          Program output   Program output
  Prediction Time        Program output   Program output
  Total Execution Time   Program output   Program output
  CPU Time               Program output   Program output
  Peak Memory            Program output   Program output

The actual values should be copied from
`MLR_KNNR_Comparison_Results.csv` after running the Python program.

## 10. Graphical Outputs

The program generates:

### MLR Actual vs Predicted

`MLR_Actual_vs_Predicted.png`

This graph compares actual concrete strength values with values
predicted by MLR.

### KNNR Actual vs Predicted

`KNNR_Actual_vs_Predicted.png`

This graph compares actual concrete strength values with KNNR
predictions.

### R² Comparison

`R2_Comparison.png`

This graph compares the R² values obtained from MLR and KNNR.

## 11. Generated Output Files

After successful execution, the following files are generated:

``` text
MLR_KNNR_Comparison_Results.csv
Actual_vs_Predicted_Values.csv
MLR_Actual_vs_Predicted.png
KNNR_Actual_vs_Predicted.png
R2_Comparison.png
```

## 12. How to Run

1.  Install Anaconda.
2.  Open Spyder.
3.  Place the Python script and Excel dataset in the same folder.
4.  Open `assignment2_mlr_knnr_spyder.py` in Spyder.
5.  Press F5 or click Run.
6.  Check the console for the MLR and KNNR results.
7.  Check the generated CSV files and graphs.

The required dataset file is:

``` text
Concrete_Compressive_Strength.xlsx
```

## 13. Interpretation

The results should be interpreted separately for each required measure:

-   Higher R² indicates greater explanatory performance.
-   Lower RMSE indicates smaller prediction error.
-   Lower MAE indicates smaller average absolute error.
-   Training and prediction times indicate computational speed.
-   CPU time and peak memory provide additional information about
    computational resource utilization.

The actual results produced by the program should be reported in the
assignment. Example values should not be substituted for the measured
results.

## 14. Conclusion

This assignment implements Multiple Linear Regression and K-Nearest
Neighbors Regression to predict concrete compressive strength. The
models are compared using R², RMSE and MAE, together with training time,
prediction time, total execution time, CPU time and peak memory usage.

The analysis demonstrates how regression models can be evaluated using
both prediction accuracy and computational performance.

## 15. Student Details

**Name:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Roll Number:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**GitHub ID:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Course:** CE46027

**Assignment:** Assignment 2

**Program:** M.Tech Geoinformatics

**Institution:** NIT Warangal
