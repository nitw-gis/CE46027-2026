Assignment 2 – Performance Comparison of MLR and KNNR

Name: Jonna Sowmya

1. Aim

The aim of this assignment is to compare the performance of:

- Multiple Linear Regression (MLR)
- K-Nearest Neighbors Regression (KNNR)
for predicting soil potassium (K) using the Smart Farming Data 2024 dataset.

2. Dataset
   
Dataset: Smart Farming Data 2024 (SF24)

The dataset contains agricultural and environmental information such as:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall
- Soil Moisture
- Soil Type
- Sunlight Exposure
- Wind Speed
- Organic Matter
- Crop Density
- Fertilizer Usage
- and other farming-related parameters

Number of rows: 2200  
Number of columns: 23

Target Variable

The target variable used for regression is:
K – Soil Potassium

The `label` column was not used as a predictor

3. Data Preprocessing

The following steps were performed:

1. Loaded the dataset using Pandas.
2. Removed the `label` column.
3. Selected `K` as the target variable.
4. Divided the data into:
   - 80% training data
   - 20% testing data
5. Standardized the input features using `StandardScaler`.

The final data sizes were:

- Training data: 1760 samples
- Testing data: 440 samples

4. Models Used

Multiple Linear Regression (MLR): used to predict soil potassium based on the available input features.
K-Nearest Neighbors Regression (KNNR): was tested with different values of K:

- K = 3
- K = 5
- K = 7
- K = 10
- K = 15
- K = 20

5. Evaluation Metrics

The models were compared using:

- R² Score – shows how well the model explains the variation in the target.
- RMSE – measures the average prediction error, with larger errors having more effect.
- MAE – measures the average absolute prediction error.
- Execution Time – measures how long the model takes to run.
- CPU Usage – observed system CPU utilization.
- RAM Usage – observed system RAM utilization.

For R², a higher value is better.
For RMSE and MAE, lower values are better.

6. Model Results

|     Model  |    R²  |   RMSE  |   MAE   | Execution Time (s) |

|     MLR    | 0.6102 | 30.5232 | 25.8786 | 0.0296 |
| KNNR (K=5) | 0.8414 | 19.4689 | 12.1591 | 0.0225 |

Best Model 
KNNR with K = 5 performed better than MLR.

It achieved:

- R² = 0.8414
- RMSE = 19.4689
- MAE = 12.1591

Compared with MLR:

- MLR R² = 0.6102
- KNNR R² = 0.8414

Therefore, KNNR provided better prediction accuracy for this dataset.

7. KNNR Comparison

Different K values were tested:

|  K |    R²  |   RMSE  |   MAE   |
|  3 | 0.8248 | 20.4644 | 11.7023 |
|  5 | 0.8414 | 19.4689 | 12.1591 |
|  7 | 0.8395 | 19.5856 | 12.3636 |
| 10 | 0.8184 | 20.8331 | 12.9523 |
| 15 | 0.8058 | 21.5472 | 13.6782 |
| 20 | 0.8059 | 21.5422 | 13.8352 |

K = 5 was selected because it gave the highest R² and lowest RMSE among the tested K values.

8. Resource Utilization

The observed system resource utilization during model execution was:

|   Model   | CPU After (%) | RAM After (%) | Execution Time (s) |
|     MLR   |     28.3      |      87.9     |        0.0296      |
| KNNR (K=5)|     39.2      |      91.6     |        0.0225      |

These CPU and RAM values represent the overall system utilization observed during execution, not the exact resources used only by the model.

9. Technologies Used

Python, Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and psutil.

10. Conclusion

KNNR with K = 5 performed better than MLR for predicting soil potassium, with a higher R² and lower RMSE and MAE. Overall, KNNR was more suitable for this dataset.
