Assignment 2: Performance Comparison of MLR and KNNR
- Name: Aswathi P (26CER1R13) PhD Scholar
  
**Dataset
- Dataset: Scrap price prediction - Multiple Linear Regression (`scrap price.csv`)
- Source: Kaggle
- Description: The car company wants to enter a new market and needs an estimation of exactly which variables affect the car prices.
**The goal is: Which variables are significant in predicting the price of a car
		       How well do those variables describe (R², RMSE, MAE) the price of a car.

- Features Used : 25 - symboling, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg
- No. of datapoints : 205

**Steps Followed
1. Downloaded the dataset (`scrap price.csv`) from Kaggle.

2. Imported required Libraries - pandas, numpy, sklearn, time, psitil, matplotlib and os

3. Loaded the dataset using pandas. Reads the csv file into a pandas dataframe called 'df'

4. Selected features (X) and the target variable (Y) - price. the features are - 'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'

5. Split the data into training (80%) and testing (20%) sets. Ensured the split is the same everytime run the code by 'random_state=42'

6. Feature scaling is done using 'StandardScaler'. KNN is distance-based. Features with large values (like curbweight) would dominate features with small values (like symboling). Scaling makes all features contribute equally.
used 'fit_transform' only on training data and 'transform' on test data (to avoid data leakage).

7. Trained Multiple Linear Regression (MLR) model.in this step recorded the starting time and memory. Trained the created model on training data. made predictions on test data.

8. Trained K-Nearest Neighbors Regression (KNNR) model. n_neighbors = 5 means the model looks 5 nearest neighbors to make prediction. measured time and memory of the process

9. Evaluated both models using R², RMSE, and MAE.

10. Measured execution time and memory usage for both models.

11. Created plots for visualization. 
	- Actual vs Predicted plots.
	- Residual plots.
	- Metrics comparison bar chart.
**Results

| Model | R²     | RMSE     | MAE      | Execution Time (s) | Memory Used (MB) |
|-------|--------|----------|----------|--------------------|------------------|
| MLR   | 0.8181 | 3789.20  | 2680.43  | 0.0779             | 3.93             |
| KNNR  | 0.7014 | 4855.44  | 2788.72  | 0.0020             | 0.15             |

**Conclusion
- Multiple Linear Regression (MLR) performed better in terms of prediction accuracy (higher R² and lower error metrics).
- K-Nearest Neighbors Regression (KNNR) was significantly faster and consumed less memory.
- For this Scrap price dataset, MLR is preferred when accuracy is the priority, while KNNR can be useful when computational speed and low resource usage are more important.
