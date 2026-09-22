# Assignment 2: Performance Comparison of Multiple Linear Regression (MLR) and K-Nearest Neighbors Regression (KNNR)

## 1. Objective

This assignment compares two regression algorithms:

- Multiple Linear Regression (MLR)
- K-Nearest Neighbors Regression (KNNR)

The models are applied to a **Crop Yield Dataset** with `Yield_ton_per_ha` as the target variable. The comparison includes prediction accuracy, execution time, and computational resource measurements.

## 2. Dataset

- **Dataset:** Crop Yield Dataset
- **Dataset file:** `crop_yield_dataset.csv`
- **Target variable:** `Yield_ton_per_ha`
- **Test size:** 20%
- **Random state:** 42
- **KNNR neighbors:** 5

The program checks the dataset dimensions, columns, data types, missing values, and duplicate rows.

If the CSV is not found beside the Python file or in the current working directory, the program opens a file-selection dialog so the user can manually select the dataset.

## 3. Software Requirements

- Python 3.12 or compatible Python version
- Spyder
- pandas
- NumPy
- Matplotlib
- scikit-learn
- psutil

Install the libraries from the Spyder IPython Console:

```python
%pip install pandas numpy matplotlib scikit-learn psutil
```

Restart the Spyder kernel after installation.

## 4. Project Structure

```text
Assignment2/
|
|-- assignment2_mlr_knnr_crop_yield.py
|-- crop_yield_dataset.csv
|
`-- assignment2_outputs/
    |-- model_comparison.csv
    |-- predictions.csv
    |-- dataset_information.csv
    |-- 01_MLR_actual_vs_predicted.png
    |-- 02_KNNR_actual_vs_predicted.png
    |-- 03_MLR_residuals.png
    |-- 04_KNNR_residuals.png
    |-- 05_R2_comparison.png
    |-- 06_RMSE_comparison.png
    |-- 07_MAE_comparison.png
    |-- 08_execution_time_comparison.png
    `-- 09_memory_change_comparison.png
```

## 5. Methodology

### 5.1 Dataset Loading

The script searches for `crop_yield_dataset.csv` in the Python script folder and the current working folder. If it is not found, a file-selection window is displayed.

The selected dataset is loaded using pandas.

### 5.2 Dataset Inspection

The program displays:

- Number of rows
- Number of columns
- Column names
- Data types
- First five records
- Missing values
- Duplicate rows

Duplicate rows are removed if they are present.

### 5.3 Feature and Target Selection

The target is:

```text
Yield_ton_per_ha
```

All remaining columns are used as predictor variables.

The script automatically identifies numerical and categorical features.

### 5.4 Missing-Value Handling

For numerical features:

- Missing values are replaced with the median.
- `StandardScaler` is applied.

For categorical features:

- Missing values are replaced with the most frequent value.
- `OneHotEncoder` converts categorical variables into numerical variables.

The same preprocessing is used for both models to maintain a fair comparison.

### 5.5 Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

with:

```text
Random state = 42
```

## 6. Multiple Linear Regression

MLR is implemented using:

```python
LinearRegression()
```

The model predicts `Yield_ton_per_ha` from the preprocessed predictor variables.

## 7. K-Nearest Neighbors Regression

KNNR is implemented using:

```python
KNeighborsRegressor(n_neighbors=5)
```

The experiment uses:

```text
K = 5
```

## 8. Accuracy Metrics

### R²

R² measures the proportion of variation in the target explained by the regression model.

### RMSE

Root Mean Squared Error measures the square root of the average squared prediction error.

### MAE

Mean Absolute Error measures the average absolute prediction error.

The three metrics are calculated for both MLR and KNNR.

## 9. Computational Resource Measurements

The program records:

- Training time
- Prediction time
- Total execution time
- Process CPU time
- Memory before execution
- Memory after execution
- Memory change

**Note:** `Memory_Change_MB` is the change in the Python process's resident memory measured before and after model execution. It is not peak RAM usage.

**Note:** `Process_CPU_Time_s` is CPU time used by the Python process. It is not system-wide CPU utilization percentage.

## 10. Graphical Outputs

The program generates nine graphs:

1. `01_MLR_actual_vs_predicted.png` — MLR actual versus predicted yield.
2. `02_KNNR_actual_vs_predicted.png` — KNNR actual versus predicted yield.
3. `03_MLR_residuals.png` — MLR residual plot.
4. `04_KNNR_residuals.png` — KNNR residual plot.
5. `05_R2_comparison.png` — R² comparison.
6. `06_RMSE_comparison.png` — RMSE comparison.
7. `07_MAE_comparison.png` — MAE comparison.
8. `08_execution_time_comparison.png` — total execution time comparison.
9. `09_memory_change_comparison.png` — memory change comparison.

## 11. Generated CSV Files

### `model_comparison.csv`

Contains:

- Model
- R²
- RMSE
- MAE
- Training time
- Prediction time
- Total execution time
- Process CPU time
- Memory before
- Memory after
- Memory change

### `predictions.csv`

Contains:

- Actual yield
- MLR predicted yield
- KNNR predicted yield

### `dataset_information.csv`

Contains dataset size, target variable, missing values, duplicate rows, train/test size, K value, feature counts, and software-version information.

## 12. How to Run in Spyder

1. Open Spyder.
2. Open `assignment2_mlr_knnr_crop_yield.py`.
3. Install the required libraries if necessary.
4. Restart the Spyder IPython Console.
5. Press **F5** or click the green **Run** button.
6. If the dataset is not found automatically, select the Crop Yield CSV from the file-selection window.
7. Wait for MLR and KNNR to finish.
8. Open the `assignment2_outputs` folder to view the generated CSV files and graphs.

## 13. Expected Results Table

After running the program, copy the values from `model_comparison.csv` into this table:

| Model | R² | RMSE | MAE | Training Time (s) | Prediction Time (s) | Total Time (s) | CPU Time (s) | Memory Change (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLR | | | | | | | | |
| KNNR (K=5) | | | | | | | | |

The execution and resource values depend on the computer and Python environment, so they should be filled using the user's actual run.

## 14. Discussion

The final discussion should be based on the generated values.

Discuss:

1. R² comparison.
2. RMSE comparison.
3. MAE comparison.
4. Training-time comparison.
5. Prediction-time comparison.
6. Total execution-time comparison.
7. Process CPU-time comparison.
8. Memory-change comparison.
9. Actual-versus-predicted graphs.
10. Residual plots.
11. Computational characteristics of MLR and KNNR.

The conclusion should be based on the measured results rather than assuming a preferred model in advance.

## 15. Reproducibility

The experiment uses:

```text
Test size    = 20%
Random state = 42
K for KNNR   = 5
```

Both models use the same train-test split and preprocessing procedure.

## 16. Limitations

- Execution time and memory measurements depend on the computer and software environment.
- Memory change is not peak RAM consumption.
- Process CPU time is not equivalent to system-wide CPU utilization.
- KNNR results depend on the selected K value.
- Results are specific to the supplied Crop Yield Dataset and preprocessing procedure.
- Dataset uniqueness should be checked separately against the department's dataset list if required.

## 17. GitHub Submission Structure

```text
Assignment2/
|
`-- <Roll_Number>/
    |-- assignment2_mlr_knnr_crop_yield.py
    |-- crop_yield_dataset.csv
    |-- README.md
    |
    `-- assignment2_outputs/
        |-- model_comparison.csv
        |-- predictions.csv
        |-- dataset_information.csv
        |-- 01_MLR_actual_vs_predicted.png
        |-- 02_KNNR_actual_vs_predicted.png
        |-- 03_MLR_residuals.png
        |-- 04_KNNR_residuals.png
        |-- 05_R2_comparison.png
        |-- 06_RMSE_comparison.png
        |-- 07_MAE_comparison.png
        |-- 08_execution_time_comparison.png
        `-- 09_memory_change_comparison.png
```

## 18. Conclusion

This assignment implements and compares Multiple Linear Regression and K-Nearest Neighbors Regression for crop-yield prediction.

The program performs data inspection, missing-value handling, categorical encoding, feature scaling, train-test splitting, model training, accuracy evaluation, computational-resource measurement, and graphical comparison.

The final conclusion should be written after the program has been executed and the actual results have been examined.

## 19. Author

**Name:** Rahul  
**Programme:** PG in Geoinformatics
