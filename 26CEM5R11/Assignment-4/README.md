# Assignment 4: ANN for Classification and Regression

**Shri Darshini (26CEM5R11)** · AI-ML (CE46027), M.Tech Geoinformatics, NIT Warangal

## Overview

Two ANNs (2+ hidden layers each), built with TensorFlow/Keras and compared with models from earlier assignments on the **same data split**:

| Part | Task | Dataset | Target | Compared with |
|---|---|---|---|---|
| A | Classification | Heart Disease (Kaggle) | `target` (0/1) | SVM, Decision Tree (Assignment 3) |
| B | Regression | Water Quality | `DissolvedOxygen (mg/L)` | MLR, KNNR (Assignment 2) |

## Files

- `Assignment-4_ANN_26CEM5R11.ipynb`: full code
- `A_classification_comparison.csv`, `B_regression_comparison.csv`: results
- `figures/`: all plots

## Part A: Classification (Heart Disease)

- 1,286 duplicate rows removed → 602 rows; 75/25 stratified split; StandardScaler
- **ANN:** 13 → 32 (ReLU) → 16 (ReLU) → 1 (Sigmoid), dropout 0.2, binary cross-entropy, Adam, early stopping

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM (linear) | 0.695 | 0.675 | 0.712 | 0.693 |
| SVM (poly) | 0.715 | 0.789 | 0.562 | 0.656 |
| SVM (RBF) | 0.735 | 0.732 | 0.712 | 0.722 |
| Decision Tree (depth 5) | **0.782** | **0.803** | 0.726 | 0.763 |
| **ANN** | 0.781 | 0.794 | **0.740** | **0.766** |

ANN ROC-AUC = **0.841**

![](figures/A_training_curves.png)
![](figures/A_confusion_roc.png)
![](figures/A_model_comparison.png)

## Part B: Regression (Water Quality)

- 6 predictors (salinity, pH, Secchi depth, water depth, water temp, air temp); missing rows removed → 1,320 rows; 80/20 split; StandardScaler
- **ANN:** 6 → 64 → 32 → 16 (ReLU) → 1 (Linear), MSE loss, Adam, early stopping

| Model | R² | RMSE (mg/L) | MAE (mg/L) |
|---|---|---|---|
| MLR | 0.363 | 1.999 | 1.519 |
| KNNR (k = 13) | **0.457** | **1.845** | 1.405 |
| **ANN** | 0.456 | 1.848 | **1.390** |

![](figures/B_training_curves.png)
![](figures/B_pred_vs_actual_residuals.png)
![](figures/B_model_comparison.png)

## Key takeaways

- **Classification:** the ANN and the Decision Tree (depth 5) are almost tied (≈ 0.78 accuracy) and both beat all SVMs. The ANN has the best recall and F1, so it misses the fewest real heart-disease cases, which matters most in a medical setting.
- **Regression:** the ANN and KNNR are practically tied (R² 0.456 vs 0.457); the ANN has the lowest MAE. Both clearly beat MLR (R² 0.363), which shows the relationship between dissolved oxygen and its drivers is non-linear.
- On small tabular datasets, the ANN gives only a small gain over simpler models and takes much longer to train. Dropout and early stopping were needed to control overfitting.
