Assignment 3 - Classification Using SVM and Decision Tree

Name: Jonna Sowmya

1. Aim

The aim of this assignment is to compare the performance of:

* Support Vector Machine (SVM)
* Decision Tree (DT)

for classifying wheat seeds into different seed types using the **Wheat Seeds dataset**.

For SVM, three different kernels were compared:

* Linear
* Polynomial
* RBF

For Decision Tree, different values of `max_depth` and `min_samples_split` were tested.

2. Dataset

**Dataset:** Wheat Seeds Dataset

The dataset contains geometric measurements of wheat kernels such as:

* Area
* Perimeter
* Compactness
* Kernel Length
* Kernel Width
* Asymmetry Coefficient
* Kernel Groove

Number of rows: 199

Number of columns: 8

Target Variable

The target variable used for classification is:

Type – Wheat Seed Type

The dataset contains three classes:

* Type 1
* Type 2
* Type 3

The class distribution was:

| Type | Samples |
| ---- | ------: |
| 1    |      66 |
| 2    |      68 |
| 3    |      65 |

3. Data Preprocessing

The following steps were performed:

1. Loaded the dataset using Pandas.
2. Selected the seven numerical features as input variables.
3. Selected `Type` as the target variable.
4. Divided the data into:

   * 80% training data
   * 20% testing data
5. Used stratified splitting to maintain the class distribution.
6. Standardized the input features using `StandardScaler`.

The final data sizes were:

* Training data: 159 samples
* Testing data:40 samples
* Input features: 7

The standardized features were used for SVM models.

Decision Tree models were trained using the original features because Decision Trees do not require feature scaling.

4. t-SNE Visualization

t-SNE was used to reduce the seven-dimensional feature space into two dimensions for visualization.

The resulting two dimensions were:

* t-SNE 1
* t-SNE 2

The points were color-coded according to the three wheat seed classes.

The t-SNE visualization was saved as:

`plots/tsne_plot.png`

t-SNE was used only for visualization and was not used for model training.

5. Models Used

Support Vector Machine (SVM)

Three SVM kernels were tested:

* Linear SVM
* Polynomial SVM
* RBF SVM

Decision Tree (DT)

Five Decision Tree configurations were tested:

| Model     | max_depth | min_samples_split |
| --------- | --------: | ----------------: |
| DT_D3_S2  |         3 |                 2 |
| DT_D5_S2  |         5 |                 2 |
| DT_D10_S2 |        10 |                 2 |
| DT_D5_S5  |         5 |                 5 |
| DT_D5_S10 |         5 |                10 |

6. Evaluation Metrics

The models were compared using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Training Time
* Prediction Time
* Memory Usage

For Accuracy, Precision, Recall, and F1-score, **higher values are better**.

For Training Time, Prediction Time, and Memory Usage, **lower values indicate better computational efficiency**.

7. Model Results

| Model          | Accuracy | Precision | Recall | F1-score | Training Time (s) | Memory Used (MB) |
| -------------- | -------: | --------: | -----: | -------: | ----------------: | ---------------: |
| Linear SVM     |   0.9250 |    0.9247 | 0.9250 |   0.9240 |          0.001818 |         0.003906 |
| Polynomial SVM |   0.8250 |    0.8863 | 0.8250 |   0.8303 |          0.001939 |         0.000000 |
| RBF SVM        |   0.8750 |    0.8744 | 0.8750 |   0.8739 |          0.001910 |         0.000000 |
| DT_D3_S2       |   0.8000 |    0.8076 | 0.8000 |   0.8009 |          0.001656 |         0.000000 |
| DT_D5_S2       |   0.8250 |    0.8409 | 0.8250 |   0.8259 |          0.002219 |         0.000000 |
| DT_D10_S2      |   0.8250 |    0.8409 | 0.8250 |   0.8259 |          0.001712 |         0.000000 |
| DT_D5_S5       |   0.8250 |    0.8409 | 0.8250 |   0.8259 |          0.001432 |         0.000000 |
| DT_D5_S10      |   0.8000 |    0.8076 | 0.8000 |   0.8009 |          0.001289 |         0.000000 |

8. Confusion Matrix

Confusion matrices were generated to examine the correct and incorrect classifications for each class.

The SVM confusion matrices were saved as:

`plots/svm_confusion_matrices.png`

The Decision Tree confusion matrices were saved as:

`plots/decision_tree_confusion_matrices.png`

 9. Best Model

Linear SVM performed the best among all tested models.

It achieved:

* Accuracy = 0.9250
* Precision = 0.9247
* Recall = 0.9250
* F1-score = 0.9240

The Linear SVM also had a very low training time of:

**0.001818 seconds**

Therefore, Linear SVM provided the best overall classification performance for this dataset.

10. Resource Utilization

The computational performance of the models was evaluated using training time and memory usage.

The measured results were:

| Model          | Training Time (s) | Memory Used (MB) |
| -------------- | ----------------: | ---------------: |
| Linear SVM     |          0.001818 |         0.003906 |
| Polynomial SVM |          0.001939 |         0.000000 |
| RBF SVM        |          0.001910 |         0.000000 |
| DT_D3_S2       |          0.001656 |         0.000000 |
| DT_D5_S2       |          0.002219 |         0.000000 |
| DT_D10_S2      |          0.001712 |         0.000000 |
| DT_D5_S5       |          0.001432 |         0.000000 |
| DT_D5_S10      |          0.001289 |         0.000000 |

The memory values represent the approximate change in process memory observed during model training.

Because the dataset contains only **199 samples**, the memory requirements were very small.

11. Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* psutil

12. Conclusion

The performance of **SVM and Decision Tree classification models** was compared using the Wheat Seeds dataset.

Three SVM kernels—**Linear, Polynomial, and RBF**—and five Decision Tree configurations were evaluated.

Among all the tested models, **Linear SVM performed the best**, achieving an accuracy of **92.5%** and an F1-score of **0.9240**.

The RBF SVM achieved the second-highest performance with an accuracy of **87.5%**, while the Decision Tree models achieved accuracies between **80% and 82.5%**.

Overall, **Linear SVM was the most suitable model for classifying the wheat seed types in this dataset**.
