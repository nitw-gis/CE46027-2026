# EuroSAT Land-Use and Land-Cover Classification Using Artificial Neural Networks and Convolutional Neural Networks

## 1. Introduction

Land-use and land-cover classification using satellite imagery is an important application of Artificial Intelligence and Machine Learning. It helps identify different geographical features such as forests, agricultural land, residential areas, rivers and industrial regions.

This project investigates the performance of two neural network approaches: Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN), using the EuroSAT RGB satellite image dataset.

The primary objective is to compare the classification performance of both approaches and understand the advantages of preserving spatial information in satellite images.

## 2. Project Objectives

- To classify satellite images into ten land-use and land-cover categories.
- To implement an Artificial Neural Network for image classification.
- To develop a Convolutional Neural Network using RGB satellite images.
- To evaluate both models using appropriate classification metrics.
- To compare the performance of ANN and CNN.
- To identify classification strengths and weaknesses across different land-cover categories.

## 3. Dataset Description

The project uses the EuroSAT RGB dataset.

| Parameter | Description |
|---|---|
| Dataset | EuroSAT |
| Total images | 27,000 |
| Image dimensions | 64 × 64 pixels |
| Channels | 3 (RGB) |
| Number of classes | 10 |
| Image format | JPEG |
| Dataset version | v6 |

### 3.1 Land-cover categories

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

### 3.2 Dataset distribution

| Dataset split | Images | Percentage |
|---|---:|---:|
| Training | 18,900 | 70% |
| Validation | 5,400 | 20% |
| Testing | 2,700 | 10% |
| Total | 27,000 | 100% |

The predefined CSV split files supplied with the dataset were used for the CNN experiment.

## 4. Artificial Neural Network (ANN)

### 4.1 Methodology

The ANN experiment was conducted using flattened satellite image pixels.

Each RGB image of dimensions 64 × 64 × 3 was converted into a one-dimensional vector containing 12,288 input values.

The experiment investigated:
- Different hidden-layer configurations, ranging from 2 to 8 layers.
- ReLU, Sigmoid and Tanh activation functions.
- Different learning rates.
- Model training and validation.
- Test evaluation and confusion matrix analysis.

### 4.2 ANN results

The previously recorded ANN test performance was:

| Metric | Result |
|---|---:|
| Test samples | 2,700 |
| Correct predictions | 1,490 |
| Test accuracy | 55.19% |

Some previously reported class recalls were:

| Class | Recall |
|---|---:|
| Forest | 94% |
| Highway | 24% |
| PermanentCrop | 30% |
| River | 28% |

The ANN showed difficulty distinguishing certain land-cover categories, particularly Highway, PermanentCrop and River.

## 5. Convolutional Neural Network (CNN)

### 5.1 Methodology

Unlike the ANN, the CNN processes satellite images in their original three-dimensional RGB structure.

The input image dimensions were maintained at 64 × 64 × 3 throughout the convolutional feature-extraction stages.

The images were normalized by dividing pixel values by 255, converting the original 0–255 range into a 0–1 range.

The labels were integer encoded from 0 to 9.

### 5.2 CNN Architecture

The following architecture was implemented:

| Layer | Configuration |
|---|---|
| Input | 64 × 64 × 3 |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 128 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | Feature vector |
| Dense | 128 neurons, ReLU |
| Dropout | 0.3 |
| Output | 10 neurons, Softmax |

### 5.3 Model parameters

| Parameter | Value |
|---|---:|
| Total parameters | 684,490 |
| Trainable parameters | 684,490 |
| Non-trainable parameters | 0 |

### 5.4 Training configuration

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Evaluation metric | Accuracy |
| Batch size | 32 |
| Maximum epochs | 25 |
| Early stopping patience | 4 |
| Best weights restoration | Enabled |

The model completed training after 13 epochs, with the best validation performance recorded at epoch 9.

## 6. CNN Training Results

| Parameter | Result |
|---|---:|
| Epochs completed | 13 |
| Training time | 235.09 seconds |
| Best validation accuracy | 84.63% |
| Lowest validation loss | 0.4316 |
| Final epoch training accuracy | 88.85% |
| Final epoch validation accuracy | 81.44% |

The model achieved its best validation accuracy of 84.63% at epoch 9.

The increase in training accuracy alongside fluctuating validation performance suggests some overfitting during later epochs. Early stopping restored the best model weights.

## 7. CNN Test Evaluation

The trained CNN was evaluated on the 2,700 test images.

| Evaluation metric | Result |
|---|---:|
| Test accuracy | 85.30% |
| Test loss | 0.4306 |
| Macro precision | 84.97% |
| Macro recall | 85.01% |
| Macro F1-score | 84.67% |
| Weighted precision | 85.74% |
| Weighted recall | 85.30% |
| Weighted F1-score | 85.18% |

## 8. Class-wise CNN Performance

| Land-cover category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| AnnualCrop | 90.91% | 86.67% | 88.74% |
| Forest | 88.92% | 99.00% | 93.69% |
| HerbaceousVegetation | 89.04% | 65.00% | 75.14% |
| Highway | 72.18% | 71.60% | 71.89% |
| Industrial | 85.87% | 97.20% | 91.18% |
| Pasture | 82.93% | 85.00% | 83.95% |
| PermanentCrop | 68.35% | 81.20% | 74.22% |
| Residential | 97.29% | 95.67% | 96.47% |
| River | 75.92% | 74.40% | 75.15% |
| SeaLake | 98.26% | 94.33% | 96.26% |

### 8.1 Interpretation

The CNN demonstrated particularly strong performance in identifying Residential, SeaLake, Forest and Industrial categories.

HerbaceousVegetation recorded a recall of 65%, indicating that some images belonging to this category were incorrectly classified.

Highway, PermanentCrop and River also presented classification challenges.

## 9. ANN vs CNN Comparative Analysis

### 9.1 Accuracy comparison

| Metric | ANN | CNN |
|---|---:|---:|
| Test accuracy | 55.19% | 85.30% |
| Test samples | 2,700 | 2,700 |
| Input representation | Flattened pixels | Original RGB images |
| Input features | 12,288 | 64 × 64 × 3 |
| Macro F1-score | Not available | 84.67% |

The CNN achieved an accuracy improvement of 30.11 percentage points over the previously reported ANN result.

### 9.2 Comparative interpretation

The ANN processes flattened pixel values and does not explicitly preserve the spatial arrangement of image features.

In contrast, the CNN uses convolutional layers to extract spatial patterns and features from satellite images.

The observed improvement suggests that convolutional feature extraction is beneficial for the EuroSAT image-classification task.

However, the exact ANN and CNN test-image membership has not yet been verified as identical. Therefore, this comparison should be interpreted as a comparison of reported experimental results rather than a strictly controlled comparison.

## 10. Limitations

- The CNN demonstrates weaker recall for HerbaceousVegetation.
- Highway, PermanentCrop and River remain comparatively difficult categories.
- The CNN exhibits signs of overfitting during later training epochs.
- The comparison with ANN requires confirmation of identical test-set membership.
- The experiment uses RGB imagery rather than all available multispectral bands.
- Further architecture optimization and data augmentation may improve performance.

## 11. Conclusion

This project investigated the application of Artificial Neural Networks and Convolutional Neural Networks for satellite image classification using the EuroSAT dataset.

The ANN achieved a previously recorded test accuracy of 55.19%, whereas the CNN achieved 85.30% test accuracy.

The CNN demonstrated stronger reported classification performance, particularly for Residential, SeaLake, Forest and Industrial categories.

The results indicate the importance of spatial feature extraction when working with satellite imagery. Convolutional Neural Networks provide an effective approach for identifying complex land-use and land-cover patterns.

Further improvements could be explored through data augmentation, deeper CNN architectures, hyperparameter optimization and multispectral image analysis.

Overall, the experiment demonstrates the potential of deep learning techniques for automated satellite image classification.

## 12. Saved Experimental Outputs

The following files were generated during the CNN experiment:

1. EuroSAT_CNN_Model.keras
2. EuroSAT_CNN_Training_History.csv
3. EuroSAT_CNN_Classification_Report.csv
4. EuroSAT_CNN_Confusion_Matrix.csv

These files preserve the trained model and its principal evaluation results for future analysis and reproducibility.