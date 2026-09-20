import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load Dataset
df = pd.read_csv("house_price.csv")

# 2. Features and Target
features = [
    'Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 
    'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality'
]
target = 'House_Price'

X = df[features]
y = df[target]

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Scaling (Essential for KNNR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Multiple Linear Regression (MLR)
start_mlr = time.perf_counter()
mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)
mlr_pred = mlr.predict(X_test_scaled)
mlr_time = time.perf_counter() - start_mlr

mlr_r2 = r2_score(y_test, mlr_pred)
mlr_rmse = np.sqrt(mean_squared_error(y_test, mlr_pred))
mlr_mae = mean_absolute_error(y_test, mlr_pred)

# 6. K-Nearest Neighbors Regression (KNNR)
start_knn = time.perf_counter()
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_time = time.perf_counter() - start_knn

knn_r2 = r2_score(y_test, knn_pred)
knn_rmse = np.sqrt(mean_squared_error(y_test, knn_pred))
knn_mae = mean_absolute_error(y_test, knn_pred)

# 7. Print Performance Comparison Table
results = pd.DataFrame({
    'Model': ['Multiple Linear Regression (MLR)', 'K-Nearest Neighbors Regression (KNNR)'],
    'R² Score': [mlr_r2, knn_r2],
    'RMSE ($)': [mlr_rmse, knn_rmse],
    'MAE ($)': [mlr_mae, knn_mae],
    'Execution Time (s)': [mlr_time, knn_time]
})

print("="*60)
print("       PERFORMANCE COMPARISON OF MLR AND KNNR")
print("="*60)
print(results.to_string(index=False))
print("="*60)