
# ANN Classification - HTRU-2 Pulsar Dataset

## Objective
To classify pulsar and non-pulsar candidates using
an Artificial Neural Network.

## Dataset
HTRU-2 Pulsar Dataset.

## Target Variable
Class:
- 0: Non-Pulsar
- 1: Pulsar

## Input Features
Eight numerical features from the HTRU-2 dataset.

## ANN Experiments
- Hidden layers: 5, 7 and 10
- Activation functions: ReLU, tanh and LeakyReLU
- Learning rates: 0.01, 0.001 and 0.0001
- Optimizer: Adam
- Loss function: Binary Cross-Entropy

Total ANN configurations: 27.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Model Comparison
The selected ANN model is compared with SVM and
Decision Tree using the same test dataset.

## Folder Structure
- Data: Input dataset
- Models: Saved trained models
- Results: Experiment results and evaluation reports
- Figures: Training plots and comparison graphs
