# Assignment 2: Performance Comparison of MLR and KNNR
This project compares the predictive performance and execution speed of **Multiple Linear Regression (MLR)** and **K-Nearest Neighbors Regression (KNNR)** using the `house_price.csv` dataset.
---
## 📌 Dataset Overview
* **Dataset Name:** House Price Prediction Dataset (`house_price.csv`)
* **Features:** `Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`
* **Target Variable:** `House_Price`
---
## 🛠️ Execution Steps Followed
1. **Data Preprocessing:** Handled missing values (if any) and checked feature distributions.
2. **Train-Test Split:** Split data into 80% training set and 20% test set with fixed `random_state=42`.
3. **Feature Scaling:** Applied `StandardScaler` to ensure features were uniformly scaled, which is crucial for distance-based algorithms like KNNR.
4. **Model Implementation:**
   - Fit **Multiple Linear Regression (MLR)** and measured execution speed.
   - Fit **K-Nearest Neighbors Regression (KNNR)** with $K=5$ and measured execution speed.
5. **Evaluation:** Evaluated accuracy using Coefficient of Determination ($R^2$), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
---
## 📊 Performance Comparison

| Model | R² Score | RMSE ($) | MAE ($) | Execution Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **Multiple Linear Regression (MLR)** | **0.9984** | **10,071.48** | **8,174.58** | **0.0035** |
| **K-Nearest Neighbors Regression (KNNR)** | 0.8916 | 83,584.74 | 69,823.26 | 0.0049 |

---
## 💡 Key Findings
* **MLR Performance:** MLR performed significantly better with an $R^2$ score of **0.9984**, showing near-linear underlying relationships in target generation.
* **KNNR Performance:** KNNR achieved an $R^2$ score of **0.8916**, showing decent fit but higher error rates due to distance calculation constraints in higher dimensions.
* **Execution Time:** MLR was marginally faster to train and predict compared to KNNR.
---
## 📁 Repository Directory Structure
```text
nitw-gis/
└── CE46027-2026/
    └── <YOUR_ROLL_NUMBER>/
        └── Assignment2/
            ├── house_price.csv
            ├── assignment2.py
            └── README.md

