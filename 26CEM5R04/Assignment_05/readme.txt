# EuroSAT Land Cover Classification Using Artificial Neural Network (ANN)

## 1. Project Overview

This project focuses on classifying satellite images into different land-cover categories using an Artificial Neural Network (ANN). The EuroSAT RGB dataset is used to train, validate, and evaluate the model.

The project investigates how different ANN configurations, activation functions, and learning rates influence land-cover classification performance.

## 2. Objectives

- To understand the application of Artificial Neural Networks in remote sensing.
- To classify satellite images into 10 land-cover categories.
- To evaluate the effect of different hidden-layer configurations.
- To compare activation functions and learning rates.
- To assess model performance using classification metrics and visualizations.

## 3. Dataset

**Dataset:** EuroSAT RGB Dataset  
**Source:** [EuroSAT Dataset – Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

The dataset contains satellite images representing 10 land-cover classes.

| Dataset | Number of Images |
|---|---:|
| Training | 18,900 |
| Validation | 5,400 |
| Testing | 2,700 |
| **Total** | **27,000** |

### Land-cover Classes

1. Annual Crop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial
6. Pasture
7. Permanent Crop
8. Residential
9. River
10. Sea Lake

## 4. Methodology

The following workflow was implemented:

1. Dataset acquisition and exploration.
2. Loading RGB satellite images.
3. Resizing images to 64 × 64 pixels.
4. Flattening image data into 12,288 input features.
5. Normalizing pixel values to the range 0–1.
6. Building and training ANN models.
7. Comparing different model configurations.
8. Evaluating the selected model using test data.
9. Generating performance tables and visualizations.

## 5. ANN Model Architecture

The ANN model uses fully connected Dense layers.

- Input features: 12,288
- Hidden layers: 2 to 8 layers (experimental comparison)
- Neurons per hidden layer: 64
- Output neurons: 10
- Output activation: Softmax
- Loss function: Categorical Crossentropy
- Optimizer: Adam

## 6. Experiments Performed

### A. Hidden-Layer Comparison
ANN configurations with 2 to 8 hidden layers were evaluated.

### B. Activation Function Comparison
The following activation functions were compared:
- ReLU
- Tanh
- Sigmoid

### C. Learning-Rate Comparison
The following learning rates were evaluated:
- 0.1
- 0.01
- 0.001
- 0.0001

## 7. Model Evaluation

The selected ANN model was evaluated using 2,700 test images.

**Final Test Accuracy: 55.19%**

Evaluation outputs include:
- Accuracy
- Loss
- Classification Report
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Actual vs. Predicted Class Comparison

## 8. Results and Visualizations

The project includes visual comparisons of:

- Hidden-layer performance
- Activation-function performance
- Learning-rate performance
- Confusion matrix
- Per-class classification performance
- Training and validation performance

The results help identify the strengths and limitations of ANN-based land-cover classification.

## 9. Tools and Technologies

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL (Pillow)

## 10. Repository Structure

```text
EuroSAT_ANN_Classification/
│
├── Notebook/
│   └── EuroSAT_ANN.ipynb
│
├── Plots/
│
├── Results_Tables/
│
├── Training_Histories/
│
├── Saved_Models/
│
└── README.md
```

## 11. Key Findings

- ANN can classify satellite images into multiple land-cover categories.
- Model performance varies with hidden-layer configuration.
- Activation functions and learning rates influence classification performance.
- The selected model achieved 55.19% test accuracy.
- Class-wise evaluation provides further insight into classification strengths and weaknesses.

## 12. Limitations

- The model uses RGB image information only.
- The ANN uses flattened image inputs and does not explicitly preserve spatial relationships between neighbouring pixels.
- Classification performance varies across land-cover categories.
- Further experimentation with CNN architectures and additional spectral bands could be explored.

## 13. Future Scope

- Compare ANN performance with Convolutional Neural Networks (CNN).
- Explore multispectral satellite imagery.
- Improve classification accuracy through hyperparameter optimization.
- Investigate class-wise errors and misclassified images.

## 14. Conclusion

This project demonstrates the application of Artificial Neural Networks for satellite image-based land-cover classification using the EuroSAT dataset. Different network configurations, activation functions, and learning rates were evaluated to understand their influence on model performance. The selected model achieved 55.19% test accuracy, providing a baseline for further improvements in remote sensing image classification.

---

**Project Category:** Remote Sensing | Geospatial AI/ML | Image Classification  
**Model:** Artificial Neural Network (ANN)  
**Dataset:** EuroSAT RGB