# Artificial Neural Network (ANN) Regression

## 1. Project overview

This project trains and evaluates a seven-hidden-layer Artificial
Neural Network (ANN) to predict concrete compressive strength.
Nine experiments compare three activation functions and three
learning rates.

## 2. Dataset

- File: `Concrete Compressive Strength.xlsx`
- Number of observations after target cleaning: 1030
- Number of predictor variables: 13
- Target variable: `strength`
- Problem type: supervised regression
- Train/test split: 80% / 20%
- Random state: 42

Predictor columns used:
- `cement`
- `slag`
- `ash`
- `water`
- `superplastic`
- `coarseagg`
- `fineagg`
- `age`
- `water_cement_ratio`
- `total_binder`
- `aggregate_to_cement`
- `cement_water_interaction`
- `age_strength_proxy`

**Feature leakage check:** The code uses every numeric column except
the target. Confirm that none of the predictors is calculated from
the target or would be unavailable when predicting a new sample.
Remove any such variable before relying on the results.

## 3. ANN architecture

| Layer | Neurons |
|---|---:|
| Input | 13 |
| Hidden 1 | 128 |
| Hidden 2 | 64 |
| Hidden 3 | 64 |
| Hidden 4 | 32 |
| Hidden 5 | 32 |
| Hidden 6 | 16 |
| Hidden 7 | 8 |
| Output | 1 |

The output is a continuous value. The ANN is implemented with
Scikit-learn `MLPRegressor` and the Adam optimizer.

## 4. Experimental design

Three activation functions:
- ReLU (`relu`)
- Hyperbolic tangent (`tanh`)
- Logistic sigmoid (`logistic`)

Three initial learning rates:
- 0.0001
- 0.001
- 0.01

This produces 3 × 3 = 9 experiments. The architecture, split,
random state, preprocessing and maximum iteration setting are held
constant across experiments.

## 5. Preprocessing

1. Load the Excel or CSV dataset.
2. Remove rows with missing target values.
3. Convert predictor columns to numeric values.
4. Remove entirely empty and constant predictor columns.
5. Fill missing predictor values with their column medians.
6. Split into training and testing subsets.
7. Standardize features using `StandardScaler`, fitted only on
   training data.

## 6. Evaluation metrics

- **R²:** coefficient of determination. Higher is generally better;
  1 is a perfect fit, 0 is equivalent to predicting the test mean,
  and a negative value indicates worse performance than that baseline.
- **RMSE:** square root of mean squared prediction error. Lower is better.
- **MAE:** mean absolute prediction error. Lower is better.

RMSE and MAE use the same unit as the target variable.

## 7. Results from all nine experiments

Experiments are sorted by test RMSE (lowest first).

| Experiment | Activation | Learning rate | Test R² | Test RMSE | Test MAE | Execution time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| ANN_relu_LR_0.01 | relu | 0.01 | 0.9154 | 4.9211 | 3.3002 | 2.868 |
| ANN_relu_LR_0.001 | relu | 0.001 | 0.9141 | 4.9573 | 3.4754 | 3.249 |
| ANN_relu_LR_0.0001 | relu | 0.0001 | 0.9064 | 5.1759 | 3.5454 | 10.401 |
| ANN_logistic_LR_0.01 | logistic | 0.01 | -0.0229 | 17.1082 | 13.7409 | 1.752 |
| ANN_tanh_LR_0.01 | tanh | 0.01 | -0.0230 | 17.1089 | 13.7411 | 2.092 |
| ANN_tanh_LR_0.001 | tanh | 0.001 | -0.0233 | 17.1119 | 13.7418 | 14.349 |
| ANN_logistic_LR_0.001 | logistic | 0.001 | -0.0234 | 17.1121 | 13.7419 | 13.761 |
| ANN_tanh_LR_0.0001 | tanh | 0.0001 | -2.5130 | 31.7052 | 26.8544 | 19.143 |
| ANN_logistic_LR_0.0001 | logistic | 0.0001 | -3.1677 | 34.5330 | 30.1063 | 14.928 |

## 8. Best model

The selected model is the one with the lowest test RMSE among the
nine experiments. This is a comparison on one held-out test split,
so it is not an unbiased final estimate after model selection.

- Experiment: ANN_relu_LR_0.01
- Activation: relu
- Learning rate: 0.01
- Training R²: 0.9758
- Testing R²: 0.9154
- Training RMSE: 2.5908
- Testing RMSE: 4.9211
- Training MAE: 1.7271
- Testing MAE: 3.3002
- Iterations: 288
- Convergence warning: False

## 9. Computational resource utilization

The following measurements are for the selected best-RMSE experiment:

| Resource | Value |
|---|---:|
| Training time (seconds) | 2.8662 |
| Total experiment time (seconds) | 2.8682 |
| CPU time (seconds) | 4.6250 |
| Average sampled CPU (%) | 161.25 |
| Maximum sampled CPU (%) | 342.80 |
| RAM before (MB) | 318.41 |
| RAM after (MB) | 318.57 |
| Peak sampled RAM (MB) | 318.59 |

CPU and RAM are process-level measurements. CPU utilization is
sampled during model fitting; peak RAM is the maximum sampled
resident memory of this Python process. Values depend on the
computer and background activity.

## 10. Output files

- `ANN_Experiment_Results.csv`: metrics and resource measurements
  for all nine experiments.
- `Best_Model_Predictions.csv`: actual, predicted and residual values.
- For each of the nine experiments: separate actual-vs-predicted,
  residual, and training-loss plots (where loss history is available).
- `BEST_Actual_vs_Predicted.png`, `BEST_Residuals_vs_Predicted.png`,
  `BEST_Residual_Distribution.png`, `BEST_Residuals_vs_Actual.png`,
  and `BEST_Training_Loss.png`: additional diagnostics for the best model.
- `ALL_Experiments_Test_R2.png`, `ALL_Experiments_Test_RMSE.png`,
  and `ALL_Experiments_Test_MAE.png`: metric comparisons across all nine.
- `README.md`: this documentation.

## 11. Interpretation and limitations

A higher test R² and lower test RMSE/MAE indicate better predictive
performance. Compare training and test metrics to assess possible
overfitting. A large gap may indicate that the model fits training
data better than unseen data.

A seven-hidden-layer network may be complex for a dataset of this
size. Some configurations may not converge within the iteration
limit; check the convergence-warning column. For a stronger
performance estimate, use repeated cross-validation or a separate
external test dataset. Since the nine configurations are compared
using the same test set, use a validation set or nested validation
for final model selection.

For this dataset when compared to multiple linear regression and KNN regresssion multiple regression model got best resluts
MLR  R2      RMSE    MAE    EXECUTION TIME   MEMORY
     0.8968  5.218   4.168  2.2809 SEC       12.2414 MB


## 12. Best model

COMPARING MLR VS KNNR VS ANNR

The best model is

- Experiment: ANN_relu_LR_0.01
- Activation: relu
- Learning rate: 0.01
- Training R²: 0.9758
- Testing R²: 0.9154
- Training RMSE: 2.5908
- Testing RMSE: 4.9211
- Training MAE: 1.7271
- Testing MAE: 3.3002
- Iterations: 288
- Convergence warning: False


## 13. Computational resource utilization



| Resource | Value |

| Training time (seconds) | 2.8662 |
| Total experiment time (seconds) | 2.8682 |
| CPU time (seconds) | 4.6250 |
| Average sampled CPU (%) | 161.25 |
| Maximum sampled CPU (%) | 342.80 |
| RAM before (MB) | 318.41 |
| RAM after (MB) | 318.57 |
| Peak sampled RAM (MB) | 318.59 |

Execution time of ANNR is 2.8662 sec slightly more than MLR 2.2809 sec
