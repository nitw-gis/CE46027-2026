# Wine Quality Classification --- SVM and Decision Tree

## 1. Project Overview

This project was completed as part of an AI/ML classification
assignment. The objective was to perform data preprocessing,
feature-space visualization, classification using Support Vector
Machines (SVM) and Decision Trees, and comparison of predictive and
computational performance.

The workflow includes:

-   Data preprocessing
-   Duplicate removal
-   80:20 stratified train-test split
-   Numerical feature standardization
-   One-hot encoding of categorical data
-   2D t-SNE feature-space visualization
-   SVM with Linear, Polynomial and RBF kernels
-   Decision Tree experiments with different `max_depth` and
    `min_samples_split` values
-   Accuracy, Precision, Recall and F1-score evaluation
-   Confusion matrices and classification reports
-   Training and prediction time measurement
-   Memory and CPU resource monitoring

## 2. Dataset

The project uses the Wine Quality dataset:

``` text
dataset/wine_quality_merged.csv
```

The dataset contains physicochemical measurements of wine and a
`quality` target.

### Input Features

**Numerical features:** - fixed acidity - volatile acidity - citric
acid - residual sugar - chlorides - free sulfur dioxide - total sulfur
dioxide - density - pH - sulphates - alcohol

**Categorical feature:** - type

**Target:** - quality

Duplicate rows were removed before the train-test split.

## 3. Tools and Libraries

-   Python
-   Jupyter Notebook
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   psutil

## 4. Project Structure

``` text
Assignment3/
├── dataset/
│   └── wine_quality_merged.csv
├── results/
│   ├── class_distribution.png
│   ├── tsne_plot.png
│   ├── model_performance_comparison.png
│   ├── training_time_comparison.png
│   ├── prediction_time_comparison.png
│   ├── memory_utilization_comparison.png
│   ├── cpu_utilization_comparison.png
│   ├── final_model_comparison.csv
│   ├── clean_performance_table.csv
│   ├── experiment_summary.txt
│   ├── confusion_matrices/
│   └── classification_reports/
└── Assignment3_Classification.ipynb
```

## 5. Data Preprocessing

### Duplicate Removal

Duplicate observations were identified and removed before model
training.

### Train-Test Split

The dataset was divided into:

-   80% training data
-   20% testing data

A fixed `random_state=42` and stratified splitting were used.

### Numerical Feature Scaling

Numerical features were standardized using `StandardScaler`.

The equation is:

$$
z = \frac{x-\mu}{\sigma}
$$

where `x` is the original value, `mu` is the mean and `sigma` is the
standard deviation.

### Categorical Encoding

The `type` feature was converted into numerical form using One-Hot
Encoding.

The preprocessing steps were included inside scikit-learn pipelines.

## 6. t-SNE Visualization

t-SNE was used to reduce the preprocessed feature space to two
dimensions for visualization.

The resulting visualization is:

``` text
results/tsne_plot.png
```

t-SNE was used only for exploratory visualization and was not used as
the input feature space for the classifiers.

## 7. Classification Models

A total of **11 models** were evaluated.

### Support Vector Machine

Three kernels were tested.

#### Linear Kernel

$$
K(\mathbf{x},\mathbf{x}') = \mathbf{x}^T\mathbf{x}'
$$

#### Polynomial Kernel

$$
K(\mathbf{x},\mathbf{x}') =
(\gamma\mathbf{x}^T\mathbf{x}' + r)^d
$$

Parameters:

-   degree = 3
-   C = 1.0
-   gamma = `scale`

#### RBF Kernel

$$
K(\mathbf{x},\mathbf{x}') =
\exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)
$$

Parameters:

-   C = 1.0
-   gamma = `scale`

### Decision Tree

The following values were tested:

-   `max_depth`: 3, 5, 10, None
-   `min_samples_split`: 2, 10

This produced:

$$
4 \times 2 = 8
$$

Decision Tree configurations.

Therefore:

``` text
3 SVM models + 8 Decision Tree models = 11 models
```

## 8. Evaluation Metrics

The models were evaluated using:

-   Accuracy
-   Weighted Precision
-   Weighted Recall
-   Weighted F1-score
-   Confusion Matrix
-   Classification Report

The F1-score is:

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

Weighted averaging was used for Precision, Recall and F1-score because
the target classes are not equally represented.

## 9. Final Experimental Results

The following table is generated directly from the uploaded output data.

  -----------------------------------------------------------------------------------
  Model            Accuracy   Precision     Recall   F1-Score   Training   Prediction
                                                                Time (s)     Time (s)
  -------------- ---------- ----------- ---------- ---------- ---------- ------------
  SVM - Linear       0.5207      0.4013     0.5207     0.4518     1.0272       0.1573

  SVM -              0.5291      0.5019     0.5291     0.4925     1.0161       0.1517
  Polynomial                                                             

  SVM - RBF          0.5385      0.4960     0.5385     0.4989     1.0356       0.4548

  Decision           0.5291      0.4800     0.5291     0.4904     0.0266       0.0075
  Tree -                                                                 
  depth=3,                                                               
  min_split=2                                                            

  Decision           0.5291      0.4800     0.5291     0.4904     0.0270       0.0090
  Tree -                                                                 
  depth=3,                                                               
  min_split=10                                                           

  Decision           0.5254      0.4827     0.5254     0.5002     0.0349       0.0079
  Tree -                                                                 
  depth=5,                                                               
  min_split=2                                                            

  Decision           0.5254      0.4827     0.5254     0.5002     0.0354       0.0073
  Tree -                                                                 
  depth=5,                                                               
  min_split=10                                                           

  Decision           0.5019      0.4903     0.5019     0.4921     0.0584       0.0079
  Tree -                                                                 
  depth=10,                                                              
  min_split=2                                                            

  Decision           0.5019      0.4860     0.5019     0.4901     0.0556       0.0090
  Tree -                                                                 
  depth=10,                                                              
  min_split=10                                                           

  Decision           0.4643      0.4612     0.4643     0.4617     0.0758       0.0081
  Tree -                                                                 
  depth=None,                                                            
  min_split=2                                                            

  Decision           0.4596      0.4532     0.4596     0.4535     0.0674       0.0080
  Tree -                                                                 
  depth=None,                                                            
  min_split=10                                                           
  -----------------------------------------------------------------------------------

## 10. Results Observations

The measured accuracy values range from **0.4596 (45.96%)** to **0.5385
(53.85%)**.

In this execution:

-   The highest observed accuracy was **0.5385 (53.85%)** for **SVM -
    RBF**.
-   The highest observed precision was **0.5019 (50.19%)** for **SVM -
    Polynomial**.
-   The highest observed weighted F1-score was **0.5002 (50.02%)** for
    **Decision Tree - depth=5, min_split=2**.
-   The fastest training time was **0.0266 seconds** for **Decision
    Tree - depth=3, min_split=2**.
-   The shortest prediction time was **0.0073 seconds** for **Decision
    Tree - depth=5, min_split=10**.

These are observations from the recorded execution. Training time,
prediction time, CPU utilization and memory measurements can vary
between runs depending on system load.

## 11. Computational Resource Analysis

The experiment measured:

-   Training time
-   Prediction time
-   Process memory usage
-   CPU utilization snapshot

Training and prediction time were measured using Python's
`time.perf_counter()`.

Memory usage was monitored using `psutil`.

The resource plots are stored in:

``` text
results/training_time_comparison.png
results/prediction_time_comparison.png
results/memory_utilization_comparison.png
results/cpu_utilization_comparison.png
```

## 12. Generated Outputs

### Visualizations

``` text
results/class_distribution.png
results/tsne_plot.png
```

### Performance Results

``` text
results/model_performance_comparison.png
results/final_model_comparison.csv
results/clean_performance_table.csv
```

### Computational Analysis

``` text
results/training_time_comparison.png
results/prediction_time_comparison.png
results/memory_utilization_comparison.png
results/cpu_utilization_comparison.png
```

### Confusion Matrices

``` text
results/confusion_matrices/
```

### Classification Reports

``` text
results/classification_reports/
```

### Experiment Summary

``` text
results/experiment_summary.txt
```

## 13. Reproducibility

The experiment uses:

``` text
Random State = 42
Test Size = 20%
SVM Models = 3
Decision Tree Models = 8
Total Models = 11
```

Using the same dataset, preprocessing procedure, model parameters and
random state should reproduce the classification metrics.

Computational measurements such as training time, prediction time,
memory usage and CPU utilization may vary slightly between executions.

## 14. How to Run

Activate the environment:

``` bash
conda activate assignment3
```

Install the required libraries:

``` bash
pip install notebook pandas numpy matplotlib seaborn scikit-learn psutil
```

Open the project directory:

``` bash
cd Assignment3
```

Launch Jupyter Notebook:

``` bash
jupyter notebook
```

Open:

``` text
Assignment3_Classification.ipynb
```

Run all notebook cells from beginning to end.

The notebook automatically generates the tables, graphs, confusion
matrices and classification reports inside the `results` directory.

## 15. Conclusion

This project implements a complete multiclass classification workflow
for wine quality prediction. It combines preprocessing, t-SNE
visualization, SVM classification using three kernels and Decision Tree
classification using multiple hyperparameter configurations.

The final experiment provides both predictive performance metrics and
computational measurements, allowing the behavior of the tested models
to be examined from multiple perspectives.

## 16. Author

**AI/ML Classification Assignment**

Dataset: Wine Quality\
Implementation: Python / Jupyter Notebook\
Machine Learning Library: Scikit-learn
