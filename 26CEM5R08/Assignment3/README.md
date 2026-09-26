SVM and Decision Tree Classification with t-SNE Visualization

1. Project Overview

This project uses the Cars dataset to classify cars into two categories based on their fuel efficiency (MPG).

The target variable is created using the following condition:

- High MPG (1): MPG ≥ 35
- Low MPG (0): MPG < 35

The project compares different Support Vector Machine (SVM) and Decision Tree (DT) models and evaluates their performance using accuracy, precision, recall, F1-score, confusion matrix, execution time, and memory usage.

A t-SNE visualization is also used to see how the car data is distributed in a two-dimensional space.

---

2. Dataset

The project uses a file named:

Cars.csv

The following four features are used for classification:

Feature| Description
HP= Horsepower of the car
VOL= Volume/size-related measurement
SP= Speed-related feature
WT= Weight of the car

The original "MPG" column is used to create the target variable.

Target Creation

The code creates a binary classification target:

df['Target'] = (df['MPG'] >= 35).astype(int)

This means:

MPG >= 35  → Target = 1 → High MPG
MPG < 35   → Target = 0 → Low MPG

So, instead of predicting the exact MPG value, the models are predicting whether a car belongs to the high-MPG or low-MPG group.

---

3. Libraries Used

The project uses the following Python libraries:

import time
import tracemalloc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

Main purpose of each library

- Pandas – loading and working with the dataset.
- NumPy – numerical operations.
- Matplotlib – creating the t-SNE plot.
- Scikit-learn – preprocessing, splitting the data, training models, and calculating evaluation metrics.
- time – measuring model execution time.
- tracemalloc – monitoring memory allocation during model training and prediction.

---

4. Step-by-Step Process

Step 1: Import the Required Libraries

First, all the required Python libraries are imported.

The display settings are also changed so that Pandas can display more columns and wider output when required.

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

---

Step 2: Load the Dataset

The "Cars.csv" file is loaded using Pandas.

df = pd.read_csv('Cars.csv')

The dataset is now stored in a Pandas DataFrame called "df".

---

Step 3: Create the Target Variable

The original dataset contains an "MPG" column.

Instead of predicting MPG directly, the project converts it into a binary classification problem.

df['Target'] = (df['MPG'] >= 35).astype(int)

The "Target" column contains only two possible values:

0 → Low MPG
1 → High MPG

This makes the problem suitable for classification algorithms such as SVM and Decision Tree.

---

Step 4: Select Input Features and Target

Four features are selected as input variables:

X = df[['HP', 'VOL', 'SP', 'WT']]

The target variable is:

y = df['Target']

Therefore:

X → HP, VOL, SP, WT
y → High MPG / Low MPG

---

5. Feature Standardization

Before training the models, the features are standardized.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Standardization changes the features so that they have approximately:

- Mean = 0
- Standard deviation = 1

This is particularly important for SVM because SVM uses distances and the scale of the features can affect the resulting decision boundary.

For example, if one feature has values in hundreds while another has values between 0 and 10, the larger-scale feature could have a stronger influence.

Standardization puts the features on a more comparable scale.

---

6. t-SNE Visualization

The project uses t-SNE (t-distributed Stochastic Neighbor Embedding) to visualize the four-dimensional feature space in two dimensions.

tsne = TSNE(
    n_components=2,
    perplexity=15,
    random_state=42
)

The transformed data is generated using:

X_tsne = tsne.fit_transform(X_scaled)

Although the original dataset has four features:

HP
VOL
SP
WT

t-SNE converts them into two dimensions:

t-SNE Dimension 1
t-SNE Dimension 2

This allows the data to be plotted on a 2D graph.

Why t-SNE is used

The purpose of this plot is mainly visualization.

It can help us visually inspect whether cars belonging to the two MPG classes form separate groups or whether they overlap.

The plot uses different labels for the two classes:

Low MPG (0)
High MPG (1)

The generated figure is saved as:

tsne_plot.png

---

7. Train-Test Split

The dataset is divided into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

The split is:

80% → Training data
20% → Testing data

The model learns patterns from the training data and is then tested on data that it has not seen during training.

"stratify=y"

The "stratify=y" parameter helps maintain a similar proportion of the two target classes in both the training and testing datasets.

"random_state=42"

Using "random_state=42" makes the split reproducible. Running the code again will produce the same split.

---

8. Machine Learning Models

Six different models are compared in this project.

8.1 Linear SVM

SVC(kernel='linear', random_state=42)

A linear SVM tries to find a straight decision boundary that separates the two classes.

In simple terms:

Low MPG | Decision Boundary | High MPG

---

8.2 Polynomial SVM

SVC(
    kernel='poly',
    degree=3,
    random_state=42
)

The polynomial kernel allows the SVM to create a more complex, curved decision boundary.

Here:

degree = 3

means a third-degree polynomial kernel is being used.

---

8.3 RBF SVM

SVC(
    kernel='rbf',
    random_state=42
)

RBF stands for Radial Basis Function.

Unlike the linear SVM, the RBF kernel can model nonlinear relationships between the input features and the target classes.

---

8.4 Shallow Decision Tree

DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

This tree is restricted to a maximum depth of 2.

A shallow tree is easier to understand and visualize, but its limited depth may prevent it from capturing more complicated patterns in the data.

---

8.5 Deep Decision Tree

DecisionTreeClassifier(
    max_depth=8,
    random_state=42
)

This tree can grow up to a depth of 8.

Because it is allowed to create more splits, it can learn more complex patterns.

However, a deeper tree can also become more sensitive to the training data and may overfit.

---

8.6 Decision Tree with Minimum Split Requirement

DecisionTreeClassifier(
    min_samples_split=5,
    random_state=42
)

Here, a node must contain at least 5 samples before it can be split.

This controls how easily the tree creates additional branches.

---

9. Measuring Execution Time and Memory

The project also measures the computational resources used by each model.

Execution time

t0 = time.perf_counter()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

t1 = time.perf_counter()

The difference between "t1" and "t0" gives the approximate time taken for training and prediction.

The result is converted into milliseconds:

(t1 - t0) * 1000

Memory usage

The code uses Python's "tracemalloc" module:

tracemalloc.start()

After training and prediction:

_, peak_mem = tracemalloc.get_traced_memory()

This records the peak Python memory allocation during the measured section.

---

10. Model Evaluation

After making predictions, several performance metrics are calculated.

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

The four main metrics are:

Accuracy

Accuracy tells us the proportion of total predictions that were correct.

Accuracy =
Correct Predictions / Total Predictions

---

Precision

Precision tells us how many of the samples predicted as High MPG were actually High MPG.

Precision =
TP / (TP + FP)

where:

- TP = True Positive
- FP = False Positive

---

Recall

Recall tells us how many of the actual High MPG cars were correctly identified.

Recall =
TP / (TP + FN)

where:

- TP = True Positive
- FN = False Negative

---

F1-Score

F1-score combines precision and recall.

F1 = 2 × (Precision × Recall)
     ---------------------------
       Precision + Recall

It is useful when we want to consider both precision and recall together.

---

11. Confusion Matrix

The confusion matrix is calculated using:

cm = confusion_matrix(y_test, y_pred)

It is then unpacked as:

tn, fp, fn, tp = cm.ravel()

The four values mean:

Value| Meaning
TN| True Negative
FP| False Positive
FN| False Negative
TP| True Positive

The output is displayed in the following order:

[TN, FP, FN, TP]

For example:

[20, 2, 1, 15]

would mean:

- 20 correctly predicted Low MPG
- 2 Low MPG samples incorrectly predicted as High MPG
- 1 High MPG sample incorrectly predicted as Low MPG
- 15 correctly predicted High MPG

---

12. Final Results Table

The results from all six models are stored in a list:

results = []

For each model, the following information is stored:

Model
Accuracy
Precision
Recall
F1-score
Execution Time
Confusion Matrix

Finally, the results are printed in an aligned table.

The output looks approximately like:

Model                | Acc    | Prec   | Rec    | F1     | Time(ms) | CM [TN,FP,FN,TP]
SVM (Linear)         | 0.xx   | 0.xx   | 0.xx   | 0.xx   | xx.xx    | [...]
SVM (Polynomial)    | 0.xx   | 0.xx   | 0.xx   | 0.xx   | xx.xx    | [...]
SVM (RBF)            | 0.xx   | 0.xx   | 0.xx   | 0.xx   | xx.xx    | [...]
DT (Shallow, d=2)   | 0.xx   | 0.xx   | 0.xx   | 0.xx   | xx.xx    | [...]
DT (Deep, d=8)      | 0.xx   | 0.xx   | 0.xx   | 0.xx   | xx.xx    | [...]
DT (min_split=5)    | 0.xx   | 0.xx   | 0.xx   | 0.xx   | xx.xx    | [...]

The actual values depend on the contents of "Cars.csv".

---

13. Project Workflow

The complete workflow can be summarized as:

Cars.csv
   ↓
Load Dataset
   ↓
Create Binary Target from MPG
   ↓
Select HP, VOL, SP and WT
   ↓
Standardize Features
   ↓
       ┌───────────────┐
       │   t-SNE       │
       │ Visualization │
       └───────────────┘
               ↓
        Train-Test Split
               ↓
       ┌───────┴────────┐
       ↓                ↓
      SVM          Decision Tree
       ↓                ↓
 Linear / Poly / RBF   d=2 / d=8 / min split=5
       └───────┬────────┘
               ↓
          Predictions
               ↓
       Model Evaluation
               ↓
 Accuracy / Precision / Recall
 F1-score / Confusion Matrix
               ↓
       Time and Memory
               ↓
          Final Results

---

14. Requirements

Install the required Python packages using:

pip install pandas numpy matplotlib scikit-learn

The "time" and "tracemalloc" modules are part of Python's standard library, so they do not need separate installation.

---

15. How to Run the Project

Step 1: Clone or download the project

Place the Python script and "Cars.csv" in the same folder.

The folder should look like:

project folder contain

Performance_comparison.py
Cars.csv
README.md

Step 2: Install the dependencies

Run:

pip install pandas numpy matplotlib scikit-learn

Step 3: Run the Python program

python Performance_comparison.py

Step 4: Check the output

The program will:

1. Load the Cars dataset.
2. Create the binary MPG target.
3. Standardize the features.
4. Generate the t-SNE visualization.
5. Split the data into training and testing sets.
6. Train six different models.
7. Make predictions.
8. Calculate evaluation metrics.
9. Measure execution time.
10. Print the final comparison table.
11. Save the t-SNE plot as "tsne_plot.png".

---

16. Conclusion

This project demonstrates a complete machine learning classification workflow using the Cars dataset.

The main purpose is not only to train a classifier, but also to compare different models and understand how their performance and computational requirements differ.

The project covers:

- Data loading and preprocessing
- Binary classification
- Feature standardization
- t-SNE dimensionality reduction for visualization
- Train-test splitting
- SVM with different kernels
- Decision Trees with different configurations
- Accuracy, precision, recall and F1-score
- Confusion matrix analysis
- Execution-time measurement
- Memory monitoring

The final results can be used to understand how different SVM kernels and Decision Tree settings behave on the same dataset.

«Note: The t-SNE plot is used for visualization only. It is not used as an input to the classification models.»