Assignment 3: Compare the performance of SVM and Decision Tree Classification
- Name: Aswathi P (26CER1R13) PhD Scholar

**Dataset
- Dataset: Wine Quality Prediction - Classification Prediction (`WineQT.csv`)
- Source: Kaggle
- Description: The dataset describes the amount of various chemicals present in wine and their effect on it's quality.
- Features Used : 12
- No. of datapoints : 1143 
- Target variable : `Quality` (classes: 3,4,5,6,7,8)
- Features used: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

**Steps Followed
1. Downloaded the Wine Quality dataset from Kaggle.
2. set up the anaconda workspace with necessary libraries
3. generate a python code for 2D t-SNE scatter plot of the feature space (color-coded by quality).
	- load dataset WineQT.csv using pandas
	- identify features (X) and target variable (Y) and droping the column ID.
	- scale the features, to make all features have mean =0 and SD =1.  so feature with larger values do not dominate the distance calculation.
	- Apply t-SNE Dimensionality reduction, in order to reduce high dimensional (12 features) into 2 dimension while trying to preserve local structure.
	- given n_components = 2 (want a 2D plot)
	- perplexity = 30 (Controls the balance between local and global structure, common value)
	- n_iter=1000 (no of iterations)
	- the output X_tsne has shape (1143,2) each wine sample now has only 2 coordinates.
	- create scatter plot. hue=y will Colors the ponts according to their quality classes. Add title and saved it as `tsne_plot.png`
4. Compared the performance of SVM and Decision tree classifier.
	- Import required libraries
	- load and prepare the data. feature (X) and Target (Y). split data into test_size 0.2 and training 0.8.	
	- scale the features. Decision tree do not need scaling, so kept as unscaled. 
	- Evaluate model by trains and make predictions. record memory and time. then calculate accuracy, precision, recall, F1- score and confusion matrix. 
	- print all results and return as summary table.
	- Tests SVM using Linear, Polynomial, and RBF kernels on scaled data.
	- Tests Decision Tree with varying max_depth, Checks how tree depth affects performance
	- Test Decision Tree with varying min_samples_split.
	- shows the final summary table.
5. Collects all results into a table. Saves it as CSV file.

**Result
================================================================================
FINAL COMPARISON SUMMARY
================================================================================
                    Model  Accuracy  Precision  Recall  F1-Score  Time (s)  Memory (MB)
             SVM (linear)    0.6114     0.5774  0.6114    0.5820    0.0582         1.89
               SVM (poly)    0.6376     0.5950  0.6376    0.6107    0.0540         0.53
                SVM (rbf)    0.6681     0.6397  0.6681    0.6420    0.0731         0.00
         DT (max_depth=3)    0.5895     0.5495  0.5895    0.5482    0.0064         0.15
         DT (max_depth=5)    0.5983     0.5621  0.5983    0.5759    0.0050         0.00
        DT (max_depth=10)    0.6245     0.6112  0.6245    0.6167    0.0080         0.00
      DT (max_depth=None)    0.6288     0.6208  0.6288    0.6245    0.0080         0.00
 DT (min_samples_split=2)    0.6288     0.6208  0.6288    0.6245    0.0080         0.00
 DT (min_samples_split=5)    0.6026     0.6015  0.6026    0.6001    0.0090         0.00
DT (min_samples_split=10)    0.5983     0.5922  0.5983    0.5922    0.0080         0.00
DT (min_samples_split=20)    0.5590     0.5475  0.5590    0.5474    0.0080         0.00

