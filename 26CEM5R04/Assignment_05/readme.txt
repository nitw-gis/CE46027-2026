# EuroSAT Land Cover Classification Using Artificial Neural Network (ANN)

## 1. Project Overview

Remote Sensing provides valuable information about the Earth's surface through satellite imagery. Satellite images are widely used for identifying land-cover features such as agricultural fields, forests, water bodies, urban areas, and industrial regions.

Land-cover classification is an important application of remote sensing because it helps transform satellite imagery into meaningful thematic information. Artificial Intelligence and Machine Learning techniques can automate this classification process and support large-scale geospatial analysis.

This project implements an **Artificial Neural Network (ANN)** to classify satellite images from the EuroSAT RGB dataset into ten different land-cover categories.

The project also investigates how changes in ANN architecture, activation functions, and learning rates influence classification performance.

The main focus is to understand the relationship between ANN configurations and their performance in satellite image classification.

---

## 2. Project Objectives

The major objectives of this project are:

- To understand the application of Artificial Neural Networks in remote sensing.
- To explore and prepare the EuroSAT satellite image dataset.
- To develop an ANN model for multi-class land-cover classification.
- To investigate the influence of different hidden-layer configurations.
- To compare the performance of ReLU, Tanh, and Sigmoid activation functions.
- To evaluate different learning rates and their influence on model training.
- To identify a suitable ANN configuration based on validation performance.
- To evaluate the selected model using independent test data.
- To interpret classification performance through statistical metrics and visualizations.

---

## 3. Dataset Description

### 3.1 EuroSAT Dataset

The EuroSAT dataset is a publicly available remote sensing image dataset developed for land-use and land-cover classification.

It contains satellite images collected from the Sentinel-2 mission, representing different land-cover and land-use categories across European geographical regions.

**Dataset source:**  
[EuroSAT Dataset – Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

**Dataset type:** RGB satellite imagery  
**Image dimensions:** 64 × 64 pixels  
**Number of spectral bands used:** 3 (Red, Green, Blue)  
**Number of classes:** 10

### 3.2 Dataset Distribution

The RGB dataset contains 27,000 images divided into training, validation, and testing subsets.

| Dataset Subset | Number of Images |
|---|---:|
| Training | 18,900 |
| Validation | 5,400 |
| Testing | 2,700 |
| **Total** | **27,000** |

- **Training dataset:** Used to train the ANN model and learn patterns from satellite images.
- **Validation dataset:** Used to monitor model performance and compare different experimental configurations.
- **Testing dataset:** Used for final evaluation of the selected model on previously unseen images.

### 3.3 Land-Cover Classes

The dataset contains the following ten land-cover categories:

| Class No. | Land-Cover Category | Description |
|---:|---|---|
| 0 | AnnualCrop | Agricultural fields containing annual crops |
| 1 | Forest | Forested regions |
| 2 | HerbaceousVegetation | Areas covered by herbaceous vegetation |
| 3 | Highway | Highways and major road infrastructure |
| 4 | Industrial | Industrial and manufacturing areas |
| 5 | Pasture | Grasslands used for grazing |
| 6 | PermanentCrop | Areas containing permanent crops |
| 7 | Residential | Residential and built-up areas |
| 8 | River | River channels and water features |
| 9 | SeaLake | Sea and lake water bodies |

---

## 4. Methodology and Workflow

The project follows a structured workflow consisting of dataset preparation, ANN model development, experimental comparison, and performance evaluation.

### 4.1 Dataset Acquisition

The EuroSAT RGB dataset was obtained from Kaggle using the KaggleHub dataset repository.

The training, validation, and testing CSV files were used to identify image filenames and their corresponding class labels.

### 4.2 Image Preprocessing

The satellite images were prepared before being supplied to the ANN model.

The preprocessing steps included:

1. Loading RGB satellite images from the dataset.
2. Converting images into RGB format.
3. Resizing images to 64 × 64 pixels.
4. Converting image pixels into numerical arrays.
5. Flattening the image arrays into one-dimensional feature vectors.
6. Normalizing pixel values by dividing them by 255.

**Image dimensions before flattening:**

64 × 64 × 3

**Number of input features:**

64 × 64 × 3 = 12,288

**Normalized pixel range:**

0 to 1

The resulting feature vectors were used as input to the ANN model.

### 4.3 ANN Model Development

An Artificial Neural Network consists of interconnected neurons arranged in layers. Each neuron receives input values, applies weights and biases, and passes the result through an activation function.

The ANN model developed in this project uses fully connected Dense layers.

The model contains:

- An input layer with 12,288 input features.
- Multiple hidden Dense layers.
- 64 neurons per hidden layer.
- An output layer containing 10 neurons.
- A Softmax output activation for multi-class classification.

The hidden-layer configurations were experimentally compared to understand how network depth influences classification performance.

### 4.4 Model Training

The ANN models were trained using the training dataset.

The validation dataset was used to monitor the model's performance during training and compare experimental configurations.

The experiments used:

- Optimizer: Adam
- Batch size: 128
- Maximum training epochs: 10
- Early stopping: Applied to monitor validation performance
- Hidden-layer neurons: 64 per layer

The model configurations were evaluated using validation loss and validation accuracy.

---

## 5. Experimental Design

Three major experimental comparisons were conducted to investigate the effect of ANN parameters on classification performance.

### 5.1 Hidden-Layer Comparison

Different ANN architectures were developed by varying the number of hidden layers.

**Configurations evaluated:**

2, 3, 4, 5, 6, 7, and 8 hidden layers.

The number of neurons was maintained at 64 per hidden layer, while other experimental settings were kept consistent.

**Purpose:**

To understand how increasing network depth influences the ability of an ANN to learn and classify land-cover patterns.

**Evaluation parameters:**

- Best epoch
- Validation loss
- Validation accuracy
- Validation error

### 5.2 Activation Function Comparison

Activation functions introduce non-linearity into neural networks and influence how information is learned and processed.

The following activation functions were compared:

| Activation Function | Description |
|---|---|
| ReLU | Introduces non-linearity by retaining positive values and setting negative values to zero |
| Tanh | Produces values between -1 and +1 |
| Sigmoid | Produces values between 0 and 1 |

**Purpose:**

To identify how different activation functions influence the learning behaviour and validation performance of the ANN model.

**Evaluation parameters:**

- Best epoch
- Validation loss
- Validation accuracy
- Validation error

### 5.3 Learning-Rate Comparison

The learning rate controls the size of the parameter updates during neural network training.

Four learning rates were evaluated:

| Experiment | Learning Rate |
|---|---:|
| 1 | 0.1 |
| 2 | 0.01 |
| 3 | 0.001 |
| 4 | 0.0001 |

**Purpose:**

To investigate how different learning rates affect the optimization process and model convergence.

**Evaluation parameters:**

- Best epoch
- Validation loss
- Validation accuracy
- Validation error

---

## 6. Final ANN Model Architecture

Based on the experimental comparison, the selected model uses six hidden layers with 64 neurons in each hidden layer.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model Type | Artificial Neural Network |
| Input Features | 12,288 |
| Hidden Layers | 6 |
| Neurons per Hidden Layer | 64 |
| Hidden Activation | ReLU |
| Output Neurons | 10 |
| Output Activation | Softmax |
| Optimizer | Adam |
| Learning Rate | 0.001 |

### Architecture

Input Layer (12,288 features)

↓

Hidden Layer 1 (64 neurons)

↓

Hidden Layer 2 (64 neurons)

↓

Hidden Layer 3 (64 neurons)

↓

Hidden Layer 4 (64 neurons)

↓

Hidden Layer 5 (64 neurons)

↓

Hidden Layer 6 (64 neurons)

↓

Output Layer (10 neurons)

The output layer produces class probabilities for the ten land-cover categories. The class with the highest predicted probability is selected as the model's predicted category.

---

## 7. Model Evaluation

The selected ANN model was evaluated using the independent test dataset containing 2,700 satellite images.

### 7.1 Evaluation Metrics

The following metrics were used to assess model performance:

**Accuracy**

Measures the proportion of correctly classified images out of the total number of test images.

**Precision**

Measures how many images predicted as a particular class actually belong to that class.

**Recall**

Measures how many images belonging to a particular class were correctly identified.

**F1-Score**

Represents the harmonic mean of precision and recall.

**Confusion Matrix**

Provides a comparison between actual and predicted classes, helping identify correct classifications