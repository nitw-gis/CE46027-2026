CE46027 Assignment 5 - 26CEM5R01

\# CE46027 - Assignment 5

\*\*Roll No: 26CEM5R01\*\*



\### 1. Dataset Handling (50k -> 2k)

Original dataset had \*\*\~50,000 rows\*\*. Due to computational limits (ANN 10-layers, CNN), we used \*\*stratified random sampling of 2,000 rows (4%)\*\*.



\- \*\*Traffic Dataset (Regression):\*\* 2000 rows, 80% Train / 20% Test

\- \*\*Gender Dataset (Classification):\*\* 2000 rows, balanced sampling (Male/Female = 50/50)



Reason: Full 50k training takes >2 hrs on CPU, 2k gives same pattern in 2-3 min.



\### 2. Models Implemented



\*\*A) Regression (To predict Traffic Stress / Volume)\*\*

| Model | Structure | Hyperparameters |

| :--- | :--- | :--- |

| \*\*MLR\*\* | Multiple Linear Regression | Baseline |

| \*\*KNNR\*\* | K-Nearest Neighbors Regressor | k=5, 10, 15 |

| \*\*ANNR\*\* | Artificial Neural Network Regressor | \*\*10-Layers\*\*, LR=\[0.001, 0.01, 0.1], Activation=\[ReLU, Tanh, Sigmoid] |



\*\*B) Classification (To predict Gender)\*\*

| Model | Structure | Hyperparameters |

| :--- | :--- | :--- |

| \*\*ANN\*\* | 10-Layer Deep ANN | LR=0.001, ReLU, Adam |

| \*\*SVM\*\* | Support Vector Machine | Kernel=Linear, RBF |

| \*\*DT\*\* | Decision Tree | Max Depth=10, 20 |



\### 3. Comparison Results



\*\*REGRESSION COMPARISON - Lower is Better\*\*

| Model | RMSE | MAE | R2 Score | Training Time |

| :--- | :--- | :--- | :--- | :--- |

| MLR | 0.452 | 0.351 | 0.68 | 1 sec |

| KNNR | 0.398 | 0.298 | 0.74 | 3 sec |

| \*\*ANNR (Best)\*\* | \*\*0.312\*\* | \*\*0.241\*\* | \*\*0.86\*\* | \*\*45 sec\*\* |



\*\*CLASSIFICATION COMPARISON - Higher is Better\*\*

| Model | Accuracy | F1-Score | Precision | Recall |

| :--- | :--- | :--- | :--- | :--- |

| SVM (RBF) | 84.2% | 0.83 | 0.84 | 0.83 |

| DT | 81.5% | 0.81 | 0.80 | 0.81 |

| \*\*ANN (Best)\*\* | \*\*91.3%\*\* | \*\*0.91\*\* | \*\*0.92\*\* | \*\*0.91\*\* |



\### 4. Which is Best \& Why?



\*\*Regression me ANNR best hai kyunki:\*\*

1\. Data non-linear hai, MLR sirf linear relation pakadta hai isliye R2 low (0.68)

2\. KNNR local pattern pakadta hai par outliers pe fail hota hai

3\. 10-Layer ANNR ne 3 activations try karke ReLU pe sabse accha non-linear mapping seekh liya - isliye RMSE sabse kam.



\*\*Classification me ANN best hai kyunki:\*\*

1\. DT overfit ho gaya (Train 95% / Test 81%)

2\. SVM linear boundary bana raha tha, data complex tha

3\. ANN 10-layers ne hierarchical features seekhe, isliye Accuracy 91.3% sabse zyada.



\### 5. How to Run

```bash

pip install pandas numpy matplotlib scikit-learn torch



python src/run\_both\_fixed.py

python src/comparison\_regression.py

python src/comparison\_gender\_classification.py



All outputs saved in `output/` and `plots/`.



\### 6. Folder Structure

\- `data/` -> Original CSVs

\- `src/run\_both\_fixed.py` -> FINAL CODE

\- `plots/` -> 6 Graphs (3 ANN LR graphs + 3 Activation graphs)

\- `output/` -> 2 Comparison CSVs + Confusion Matrices

