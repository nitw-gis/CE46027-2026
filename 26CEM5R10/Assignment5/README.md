# Assignment 5: ANN for Regression and Classification

**Course:** CE46027 | **Roll number:** 26CEM5R10 | **Name:** Sahla Abdul Salam

## Objective
Build Artificial Neural Networks (Multi-Layer Perceptrons) on the two datasets used in earlier assignments and study how
(1) the number of hidden layers, (2) the activation function and (3) the learning rate affect the error.

## Datasets (both from Kaggle)
| Task | Dataset | Target | Earlier assignment |
|---|---|---|---|
| Regression | Corn dataset (`corn_data.csv`, 422 rows) | `Yield` | Assignment 2 (MLR vs KNNR) |
| Classification | Mobile Price Classification (`train.csv`, 2000 rows, 4 classes) | `price_range` | Assignment 3 (SVM vs Decision Tree) |

## Steps followed
1. Loaded both CSV files in Google Colab (notebook `Assignment5_ANN.ipynb`).
2. Preprocessing: dropped the `Farmer` ID column from corn, filled missing values (median for numbers, most frequent for text), one-hot encoded text columns.
3. Split the data 80% train / 20% test (stratified for classification) and standardised features (and the corn target) using the training set only.
4. Built ANNs with scikit-learn `MLPRegressor` and `MLPClassifier` (Adam optimiser, 64 neurons per hidden layer, 300 epochs).
5. Ran three experiments, changing one parameter at a time while the others stay at the baseline (2 hidden layers, ReLU, learning rate 0.001):
   - Hidden layers: 1, 2, 3, 4
   - Activation function: ReLU, tanh, logistic (sigmoid)
   - Learning rate: 0.0001, 0.001, 0.01, 0.1
6. Also ran a grid of activation x learning rate (heatmap).
7. Recorded train/test error and training time, and plotted loss curves, error bars and training time.

## Error metrics
- Regression: RMSE (MAE and R2 also saved in the CSV)
- Classification: error rate = 1 - accuracy (accuracy and weighted F1 also saved in the CSV)

## Files
- `Assignment5_ANN.ipynb`: complete code with outputs
- `results/`: all plots and result tables
  - `regression_hidden_layers.png`, `regression_activation.png`, `regression_learning_rate.png`
  - `classification_hidden_layers.png`, `classification_activation.png`, `classification_learning_rate.png`
  - `regression_heatmap_activation_lr.png`, `classification_heatmap_activation_lr.png`
  - `comparison_regression_vs_classification.png`
  - `regression_results.csv`, `classification_results.csv`
- `corn_data.csv`, `train.csv`: datasets

## Results

**Test error (corn RMSE / mobile price error rate)**

| Parameter | Value | Corn RMSE | Mobile error rate |
|---|---|---|---|
| Hidden layers | 1 | 73.6 | 7.25% |
| | 2 | 75.8 | 8.75% |
| | 3 | 74.8 | 7.00% |
| | 4 | 77.2 | 9.50% |
| Activation | ReLU | 75.8 | 8.75% |
| | tanh | 59.7 | 5.25% |
| | logistic | 48.7 | 2.25% |
| Learning rate | 0.0001 | 77.6 | 11.75% |
| | 0.001 | 75.8 | 8.75% |
| | 0.01 | 61.9 | 7.50% |
| | 0.1 | 137.3 | 9.75% |

## Observations
- **Hidden layers:** Adding layers did not meaningfully reduce test error for either dataset. For corn, training RMSE fell from about 24 (1 layer) to about 7 (4 layers) while test RMSE stayed around 74-77, which shows overfitting on a small dataset. For mobile price, training error was already 0 with every depth, and test error stayed between 7% and 9.5%. For corn, training time increased with depth.
- **Activation function:** Logistic gave the lowest test error on both datasets (corn RMSE 48.7, mobile error 2.25%), followed by tanh, with ReLU the worst at the baseline learning rate. Logistic was also slower to train on the mobile dataset.
- **Learning rate:** 0.0001 was too small, so the loss was still high after 300 epochs (corn RMSE 77.6, mobile error 11.75%). 0.1 was too large and unstable, especially for regression (RMSE 137.3). The best learning rate depended on the activation function (heatmap): for corn, ReLU was best at 0.01, tanh at 0.0001 and logistic at 0.001.
- **Regression vs classification:** The corn regression was much more sensitive to the learning rate and activation than the mobile price classification. Corn has only about 420 rows, so it overfits more easily, while the mobile classification stayed in a narrow error range (about 2% to 12%) across all settings.

## Conclusion
In these experiments, a logistic (sigmoid) activation with a learning rate of 0.001 and 2 hidden layers gave the lowest corn test RMSE (48.7), and logistic gave the lowest mobile price error (2.25%). More hidden layers did not help and led to overfitting on the small corn dataset, while a learning rate of 0.1 was clearly too high. Each setting was trained once with a fixed random seed on a small test set, so small differences between settings may be partly due to chance.

## How to run
Open `Assignment5_ANN.ipynb` in Google Colab, upload `corn_data.csv` and `train.csv` when asked, and run all cells. In the notebook the mobile file may be named `train (1).csv`, so set `CLS_CSV` to the name of the uploaded file.
