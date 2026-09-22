# Assignment 2: Performance Comparison of MLR and KNNR

## 1. Project Overview

This project compares the performance of two regression machine-learning models:

- Multiple Linear Regression (MLR)
- K-Nearest Neighbors Regression (KNNR)

The dataset used is `FINAL_USO.csv` / `FINAL_USO(1).csv`.

The Python program is designed to run in **Spyder** and automatically performs data loading, preprocessing, model training, prediction, performance evaluation, result saving, and graph generation.

The Python script used for this project is `assignment2_MLR_KNNR_FINAL_FIXED.py`.

---

## 2. Dataset

### Dataset Name
`FINAL_USO.csv`

The program also recognizes:

`FINAL_USO(1).csv`

### Target Variable
`Adj Close`

The `Adj Close` column is used as the dependent/target variable.

### Predictor Variables
The program removes:

- `Adj Close` — target variable
- `Date` — excluded because it is not directly used as a numerical predictor

The remaining numerical columns are used as predictor variables.

---

## 3. Software Requirements

The following software is required:

- Python 3.x
- Spyder IDE
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Optional:

- psutil — used for memory-usage measurement

---

## 4. Required Python Packages

Install the packages using Anaconda Prompt or the Spyder/IPython console.

```bash
pip install numpy pandas matplotlib scikit-learn psutil
```

If you are using Anaconda:

```bash
conda install numpy pandas matplotlib scikit-learn psutil
```

---

## 5. Project Folder Structure

Keep the files in the following structure:

```text
Assignment_2/
│
├── assignment2_MLR_KNNR_FINAL_FIXED.py
├── FINAL_USO.csv
│
└── results/
```

The `results` folder is created automatically by the program.

If the CSV file is not in the same folder as the Python script, the program opens a file-selection window so that you can manually select the dataset.

---

## 6. How to Run in Spyder

### Step 1
Open **Spyder**.

### Step 2
Open:

```text
assignment2_MLR_KNNR_FINAL_FIXED.py
```

### Step 3
Make sure the required Python packages are installed.

### Step 4
Click the green **Run ▶** button.

### Step 5
If the program cannot automatically find the CSV file, a file-selection window will appear.

Select:

```text
FINAL_USO.csv
```

or

```text
FINAL_USO(1).csv
```

### Step 6
The program will execute the complete analysis.

---

## 7. Data Preprocessing

The program performs the following preprocessing steps:

1. Loads the CSV dataset.
2. Displays the first five rows.
3. Displays dataset information.
4. Checks missing values.
5. Checks duplicate rows.
6. Removes duplicate rows if they exist.
7. Separates the target variable `Adj Close`.
8. Removes the `Date` column when available.
9. Keeps numerical predictor variables.
10. Uses a train-test split.

### Train-Test Split

The dataset is divided into:

- Training data: 80%
- Testing data: 20%
- Random state: 42

---

## 8. Machine Learning Models

### 8.1 Multiple Linear Regression

Multiple Linear Regression is used to model the relationship between the target variable and multiple numerical predictor variables.

The model used in the program is:

```python
LinearRegression()
```

### 8.2 K-Nearest Neighbors Regression

KNN Regression predicts the target based on nearby observations.

The program uses:

```python
KNeighborsRegressor(n_neighbors=5)
```

Therefore:

```text
K = 5
```

---

## 9. Machine Learning Pipeline

Both models are processed using a Scikit-learn pipeline.

The pipeline performs:

1. Missing-value imputation using the median.
2. Feature standardization using `StandardScaler`.
3. Regression model training.

The pipeline structure is:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", model)
])
```

---

## 10. Performance Parameters

The program calculates the following performance measures.

### R² Score

R² measures how well the model explains the variation in the target variable.

Higher R² generally indicates that more variation in the target is explained by the model.

### RMSE

Root Mean Squared Error measures the magnitude of prediction errors.

Lower RMSE indicates smaller prediction errors.

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted values.

Lower MAE indicates smaller average prediction errors.

---

## 11. Computational Performance

The program also records:

- Execution time
- CPU time
- Memory change

Memory usage is measured using the optional `psutil` package.

If `psutil` is not installed, the program continues to run and reports memory usage as unavailable.

---

## 12. Generated Output Files

After successful execution, a `results` folder is created.

The following files are generated:

```text
results/
│
├── model_comparison_results.csv
├── dataset_information.csv
├── MLR_Actual_vs_Predicted.png
├── KNNR_Actual_vs_Predicted.png
├── R2_Comparison.png
├── RMSE_Comparison.png
├── MAE_Comparison.png
├── Execution_Time_Comparison.png
└── Memory_Utilization_Comparison.png
```

The memory graph is generated when `psutil` is available.

---

## 13. Description of Generated Graphs

### MLR_Actual_vs_Predicted.png

Shows actual `Adj Close` values against values predicted by Multiple Linear Regression.

### KNNR_Actual_vs_Predicted.png

Shows actual `Adj Close` values against values predicted by KNN Regression.

### R2_Comparison.png

Compares the R² scores of MLR and KNNR.

### RMSE_Comparison.png

Compares RMSE values of MLR and KNNR.

### MAE_Comparison.png

Compares MAE values of MLR and KNNR.

### Execution_Time_Comparison.png

Compares the model execution times.

### Memory_Utilization_Comparison.png

Compares memory changes during model execution when `psutil` is available.

---

## 14. Output CSV Files

### model_comparison_results.csv

Contains:

- Model
- R²
- RMSE
- MAE
- Execution Time
- CPU Time
- Memory Change

### dataset_information.csv

Contains:

- Dataset name
- Number of rows
- Number of columns
- Target variable
- Number of features
- Training samples
- Testing samples
- Training percentage
- Testing percentage
- Random state
- K value

---

## 15. Expected Program Flow

```text
Start
  ↓
Find / Select CSV Dataset
  ↓
Load Dataset
  ↓
Display Dataset Information
  ↓
Check Missing Values
  ↓
Check and Remove Duplicates
  ↓
Select Adj Close as Target
  ↓
Remove Date
  ↓
Select Numerical Features
  ↓
Split Data into 80% Training and 20% Testing
  ↓
Preprocess Data
  ↓
Train Multiple Linear Regression
  ↓
Train KNN Regression
  ↓
Predict Test Data
  ↓
Calculate R², RMSE and MAE
  ↓
Calculate Execution and CPU Time
  ↓
Save Results
  ↓
Generate Graphs
  ↓
Assignment Completed
```

---

## 16. Important Notes

- Do not rename the target column `Adj Close`.
- The CSV file must be a valid CSV dataset.
- Keep the Python script and CSV file in an easily accessible folder.
- If the CSV file is stored elsewhere, use the file-selection window.
- Do not close the graph windows while the program is still generating the remaining graphs.
- The `results` folder is created automatically.

---

## 17. Troubleshooting

### Error: FINAL_USO.csv not found

Place the CSV file in the same folder as the Python script, or select it using the file-selection window.

### Error: No module named pandas

Install Pandas:

```bash
pip install pandas
```

### Error: No module named sklearn

Install Scikit-learn:

```bash
pip install scikit-learn
```

### Error: No module named matplotlib

Install Matplotlib:

```bash
pip install matplotlib
```

### Error: No module named psutil

Install psutil:

```bash
pip install psutil
```

The program can still run without psutil, but memory usage will not be measured.

---

## 18. Conclusion

This assignment implements and compares Multiple Linear Regression and K-Nearest Neighbors Regression using the `FINAL_USO` dataset.

The program evaluates the models using statistical performance measures such as R², RMSE, and MAE, along with computational measures such as execution time, CPU time, and memory change. The results are saved as CSV files and graphical comparisons are generated automatically.

---

## Author

**Assignment 2 – Performance Comparison of MLR and KNNR**

**Environment:** Python / Spyder

**Dataset:** FINAL_USO.csv

**Target:** Adj Close
