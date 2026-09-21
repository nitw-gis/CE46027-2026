# Multiple Linear Regression vs KNN Regression

## 1. Project Title

**Comparison of Multiple Linear Regression (MLR) and K-Nearest Neighbours Regression (KNNR) for Real Estate Price Prediction**

---

## 2. Project Description

This project compares the performance of two regression algorithms:

1. Multiple Linear Regression (MLR)
2. K-Nearest Neighbours Regression (KNNR)

The models are applied to the **Real estate.csv** dataset to predict the **house price of unit area**.

The performance of both models is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² (Coefficient of Determination)

The program also produces graphs for model comparison and saves the results in a `results` folder.

---

## 3. Dataset

### Dataset name

`Real estate.csv`

### Number of observations

414 records

### Number of columns

8 columns

### Variables

| Column | Description | Role |
|---|---|---|
| No | Serial/ID number | Removed |
| X1 transaction date | Transaction date | Predictor |
| X2 house age | Age of house | Predictor |
| X3 distance to the nearest MRT station | Distance to nearest MRT station | Predictor |
| X4 number of convenience stores | Number of nearby convenience stores | Predictor |
| X5 latitude | Latitude | Predictor |
| X6 longitude | Longitude | Predictor |
| Y house price of unit area | House price per unit area | Target |

The `No` column is removed because it is only an identification/serial-number field and does not represent a meaningful predictive feature.

---

## 4. Software Requirements

The program can be executed using:

- Python 3.x
- Anaconda
- Spyder
- Jupyter Notebook

---

## 5. Required Python Libraries

The following libraries are used:

```text
pandas
numpy
matplotlib
scikit-learn
```

Install them using Anaconda Prompt:

```bash
conda activate mlr_knnr
conda install pandas numpy matplotlib scikit-learn -y
```

---

## 6. Project Folder Structure

Keep the files in the following structure:

```text
MLR_KNN_Project/
│
├── MLR_KNN_Real_Estate_Analysis.py
├── Real estate.csv
│
└── results/
    ├── Model_Performance_Comparison.csv
    ├── Actual_vs_Predicted.csv
    ├── MLR_Actual_vs_Predicted.png
    ├── KNN_Actual_vs_Predicted.png
    ├── R2_Comparison.png
    ├── MAE_Comparison.png
    ├── RMSE_Comparison.png
    └── K_vs_CV_R2.png
```

The `results` folder is automatically created by the Python program.

---

## 7. How to Run the Program

### Step 1: Activate the environment

Open Anaconda Prompt and enter:

```bash
conda activate mlr_knnr
```

### Step 2: Open Spyder

Enter:

```bash
spyder
```

Alternatively, open Spyder from Anaconda Navigator.

### Step 3: Open the Python file

Open:

```text
MLR_KNN_Real_Estate_Analysis.py
```

### Step 4: Check the dataset

Make sure:

```text
Real estate.csv
```

is in the same folder as the Python source file.

The program uses:

```python
file_path = "Real estate.csv"
```

Therefore, no Windows absolute path is required.

### Step 5: Run the program

In Spyder:

```text
Run → Run
```

or press:

```text
F5
```

The program will display the results in the console and generate the graphs.

---

## 8. Methodology

The complete workflow is:

```text
Real estate.csv
       |
       v
Load dataset
       |
       v
Check data structure
       |
       v
Check missing values and duplicates
       |
       v
Select predictor variables and target
       |
       v
Remove ID column
       |
       v
Train-Test Split
(80% Training / 20% Testing)
       |
       +-----------------------+
       |                       |
       v                       v
Multiple Linear          KNN Regression
Regression                    |
       |                 Standard Scaling
       |                       |
       |                 Find suitable K
       |                       |
       +-----------+-----------+
                   |
                   v
             Predictions
                   |
                   v
          Performance Metrics
                   |
       +-----------+-----------+
       |           |           |
       v           v           v
      MAE         MSE         RMSE
                   |
                   v
                  R²
                   |
                   v
          Model Comparison
                   |
                   v
               Graphs
```

---

## 9. Multiple Linear Regression

Multiple Linear Regression predicts the target variable using multiple independent variables.

The general equation is:

```text
Y = b0 + b1X1 + b2X2 + ... + bnXn
```

For this project, the model uses:

```text
Transaction date
House age
Distance to nearest MRT station
Number of convenience stores
Latitude
Longitude
```

to predict:

```text
House price of unit area
```

The Python implementation uses:

```python
LinearRegression()
```

---

## 10. K-Nearest Neighbours Regression

KNN Regression predicts a value based on nearby observations.

The basic process is:

```text
New observation
       |
       v
Calculate distances
       |
       v
Find nearest observations
       |
       v
Select K neighbours
       |
       v
Calculate prediction
```

Because KNN is distance-based, the predictor variables are standardized using:

```python
StandardScaler()
```

The program tests K values from:

```text
K = 1 to K = 20
```

using 5-fold cross-validation.

The selected K value is then used for the final KNN model.

---

## 11. Train-Test Split

The dataset is divided into:

```text
80% → Training data
20% → Testing data
```

The code uses:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The `random_state=42` ensures that the same split can be reproduced.

---

## 12. Performance Metrics

### 12.1 Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

```text
MAE = average(|Actual - Predicted|)
```

Lower MAE indicates smaller prediction errors.

---

### 12.2 Mean Squared Error (MSE)

MSE calculates the average squared prediction error.

```text
MSE = average((Actual - Predicted)^2)
```

Lower MSE indicates smaller errors.

---

### 12.3 Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

```text
RMSE = sqrt(MSE)
```

Lower RMSE indicates smaller prediction errors.

---

### 12.4 R² Score

R² measures the proportion of variation in the target variable explained by the model.

```text
R² = 1 - (Sum of Squared Errors / Total Sum of Squares)
```

A higher R² generally indicates that the model explains more of the variation in the target variable.

---

## 13. Output Files

The program automatically creates a `results` folder.

### 13.1 Model Performance Comparison

```text
Model_Performance_Comparison.csv
```

Contains:

```text
Model
MAE
MSE
RMSE
R2
```

for both MLR and KNN.

---

### 13.2 Actual vs Predicted Values

```text
Actual_vs_Predicted.csv
```

Contains:

```text
Actual
MLR Predicted
KNN Predicted
```

---

### 13.3 MLR Actual vs Predicted Graph

```text
MLR_Actual_vs_Predicted.png
```

Shows actual house prices against MLR predicted values.

---

### 13.4 KNN Actual vs Predicted Graph

```text
KNN_Actual_vs_Predicted.png
```

Shows actual house prices against KNN predicted values.

---

### 13.5 R² Comparison

```text
R2_Comparison.png
```

Compares the R² values of MLR and KNN.

---

### 13.6 MAE Comparison

```text
MAE_Comparison.png
```

Compares the MAE values of MLR and KNN.

---

### 13.7 RMSE Comparison

```text
RMSE_Comparison.png
```

Compares the RMSE values of MLR and KNN.

---

### 13.8 K Value vs Cross-Validation R²

```text
K_vs_CV_R2.png
```

Shows the relationship between the K value and 5-fold cross-validation R².

---

## 14. Expected Console Output

After successful execution, the console will display information similar to:

```text
============================================================
DATASET INFORMATION
============================================================

Dataset shape:
(414, 8)

...

============================================================
MULTIPLE LINEAR REGRESSION
============================================================

MLR Performance:
MAE  : ...
MSE  : ...
RMSE : ...
R2   : ...

============================================================
KNN REGRESSION
============================================================

Best K value: ...

KNN Performance:
MAE  : ...
MSE  : ...
RMSE : ...
R2   : ...

============================================================
MODEL PERFORMANCE COMPARISON
============================================================
```

The exact metric values are calculated when the program is executed and should be reported from the program output.

---

## 15. Interpretation of Results

For the comparison:

- **Lower MAE** indicates lower average absolute prediction error.
- **Lower MSE** indicates lower squared prediction error.
- **Lower RMSE** indicates lower prediction error in the target variable's units.
- **Higher R²** indicates more variation in the target variable is explained by the model.

The results should be interpreted using the actual values generated from the test dataset.

---

## 16. Advantages of the Approach

### Multiple Linear Regression

- Simple and interpretable.
- Shows the relationship between predictors and target.
- Provides regression coefficients.
- Computationally efficient.

### KNN Regression

- Non-parametric regression method.
- Can model nonlinear relationships.
- Does not require a predefined mathematical relationship between predictors and target.
- Uses neighbouring observations to make predictions.

---

## 17. Important Notes

1. Do not use the `No` column as a predictor.
2. KNN requires feature scaling because it is distance-based.
3. Do not compare models using training performance alone.
4. Use the test-set predictions for the final MAE, MSE, RMSE and R² comparison.
5. The K value is selected using cross-validation.
6. Keep `Real estate.csv` in the same directory as the Python file unless the `file_path` variable is changed.
7. If the CSV filename is changed, update:

```python
file_path = "Real estate.csv"
```

---

## 18. Troubleshooting

### Error: FileNotFoundError

If you get:

```text
FileNotFoundError: Real estate.csv
```

make sure the CSV is in the same folder as the Python file.

Alternatively, use a full path with forward slashes:

```python
file_path = "D:/NIT/Github/Real estate.csv"
```

Avoid writing Windows paths like:

```python
file_path = "D:\NIT\Github\Real estate.csv"
```

because Python may interpret backslashes as escape characters.

---

### Error: ModuleNotFoundError

For example:

```text
ModuleNotFoundError: No module named 'sklearn'
```

activate the environment and install the packages:

```bash
conda activate mlr_knnr
conda install pandas numpy matplotlib scikit-learn -y
```

---

## 19. Conclusion

This project implements and compares Multiple Linear Regression and K-Nearest Neighbours Regression for real-estate price prediction. The dataset is divided into training and testing subsets, and both models are evaluated using MAE, MSE, RMSE and R². KNN uses feature standardization and cross-validation to select an appropriate K value. The program produces numerical comparison results, actual-versus-predicted tables, and graphical comparisons for interpretation.
