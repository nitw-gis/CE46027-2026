#  Cars Dataset Classification & Neural Network Optimization

This repository contains a robust machine learning pipeline designed to predict fuel efficiency category (`MPG`) using vehicle performance characteristics. It implements and compares multiple supervised classification models—including **Support Vector Machines (SVM)**, **Decision Trees (DT)**, and a heavily optimized **Artificial Neural Network (ANN)** tuned via a grid search across 18 distinct hyperparameter configurations.

---

## 📂 Project Structure & File Manifest

* **`Cars.csv`**: The primary dataset containing vehicle features (`HP`, `VOL`, `SP`, `WT`) and miles per gallon (`MPG`).
* **`cars_full_comparison.py`**: The end-to-end Python script handling data loading, feature scaling, target transformation, ANN grid search optimization, traditional model training, and performance evaluation.
* **`Cars_Classification_Comparison.csv`**: Exported comparative metrics file containing accuracy, precision, recall, F1-scores, training/inference time, and confusion matrices.

---

## ⚙️ Methodology & Pipeline Workflow

1. **Target Engineering**: 
   The continuous target variable (`MPG`) is converted into a binary classification problem by setting a threshold: cars with an $MPG \ge 35$ are labeled as high efficiency (`1`), while those below are labeled as low efficiency (`0`).

2. **Feature Scaling**: 
   All continuous input features (`HP`, `VOL`, `SP`, `WT`) are standardized using `StandardScaler` to ensure zero mean and unit variance, which stabilizes gradient descent and distance-based calculations.

3. **ANN Hyperparameter Grid Search (18 Experiments)**: 
   The script systematically evaluates **18 unique ANN architectures** using combinations of:
   * **Activation Functions (3)**: `relu`, `tanh`, `elu`
   * **Learning Rates (3)**: `0.1`, `0.01`, `0.001`
   * **Hidden Layer Architectures (2)**: `(32, 16)` across two dense layers
   * *Early stopping* with a validation split is utilized to prevent overfitting and determine the optimal epoch count.

4. **Model Benchmarking**: 
   The winning ANN configuration is retrained on the full training dataset and benchmarked side-by-side with diverse configurations of SVMs (Linear, Polynomial, RBF) and Decision Trees (varying depth and split criteria).

---

## 📊 Performance Benchmarks & Results

| Model | Accuracy | Precision | Recall | F1-Score | Time (ms) | Confusion Matrix `[TN, FP, FN, TP]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned ANN** | 0.88 | 0.82 | 1.00 | 0.90 | 851.78 | `[6, 2, 0, 9]` |
| **SVM (Linear)** | 0.88 | 0.82 | 1.00 | 0.90 | 1.15 | `[6, 2, 0, 9]` |
| **SVM (Polynomial)** | 0.76 | 0.69 | 1.00 | 0.82 | 0.90 | `[4, 4, 0, 9]` |
| **SVM (RBF)** | 0.88 | 0.82 | 1.00 | 0.90 | 0.85 | `[6, 2, 0, 9]` |
| **DT (Shallow, d=2)** | 0.88 | 0.89 | 0.89 | 0.89 | 0.70 | `[7, 1, 1, 8]` |
| **DT (Deep, d=8)** | 0.88 | 0.89 | 0.89 | 0.89 | 0.66 | `[7, 1, 1, 8]` |
| **DT (min_split=5)** | 0.88 | 0.89 | 0.89 | 0.89 | 0.67 | `[7, 1, 1, 8]` |

---

## 💡 Key Findings & Detailed Interpretation

* **Recall Perfection**: The Tuned ANN, SVM (Linear), and SVM (RBF) achieved a **Recall of 1.00**, meaning they successfully caught every single positive instance with zero false negatives ($FN = 0$).
* **Confusion Matrix Analysis**: For ANN and top SVM variants, out of 17 test samples, they correctly identified all 9 high-MPG cars ($TP = 9$) and 6 low-MPG cars ($TN = 6$), with only 2 false positives ($FP = 2$).
* **Efficiency vs. Performance**: While traditional models like SVM and Decision Trees execute in **under 2 milliseconds**, the Tuned ANN requires ~850 ms due to iterative gradient descent optimization and multiple hidden layer computations.
* **Decision Tree Stability**: Modifying tree depth (`d=2` vs `d=8`) or minimum sample split parameters showed robust performance consistency across configurations on this dataset scale.

---

## 🚀 Getting Started & Execution

### Prerequisites
Make sure you have Python installed along with the required machine learning and deep learning libraries:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

### Running the Script
Place `Cars.csv` in your working directory and execute the script via your terminal, command prompt, or IDE (such as Spyder):
```bash
python cars_full_comparison.py
```