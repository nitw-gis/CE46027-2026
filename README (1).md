# ANN Stellar Classification --- Parameter Experimentation

## Project Overview

This project evaluates an Artificial Neural Network (ANN) across
multiple hyperparameter combinations for a classification task. The
experiments compare different numbers of hidden layers, activation
functions, and learning rates.

The results are stored in a CSV file and visualized using five plots.

## Hyperparameters

-   **Hidden layers:** 2 to 10
-   **Activation functions:** `relu`, `tanh`, `sigmoid`
-   **Learning rates:** `0.0001`, `0.001`, `0.01`

This gives **81 parameter combinations** (9 hidden-layer settings × 3
activation functions × 3 learning rates).

Other training settings, such as epochs, batch size, neuron count,
optimizer, and early stopping, depend on the experiment code.

## Recorded Results

The results CSV contains these columns:

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `Hidden_Layers`                     Number of hidden layers used

  `Activation`                        Activation function used

  `Learning_Rate`                     Learning rate used

  `Validation_Accuracy`               Accuracy measured on validation
                                      data

  `Validation_Error`                  Validation error recorded by the
                                      experiment

  `Validation_Error_Percent`          Validation error expressed as a
                                      percentage

  `Validation_Log_Loss`               Log-loss on validation data

  `Validation_Weighted_F1`            Weighted F1 score on validation
                                      data

  `Training_Time_Seconds`             Training time in seconds
  -----------------------------------------------------------------------

## Visualizations

The plotting script creates five figures:

1.  **Training Time for All ANN Experiments** --- compares training time
    across experiments.
2.  **Validation Accuracy for All ANN Experiments** --- compares
    validation accuracy across experiments.
3.  **Effect of Hidden Layers on Training Time** --- shows how training
    time varies with hidden-layer count.
4.  **Validation Accuracy by Activation Function** --- compares
    validation accuracy for each activation function.
5.  **Validation Accuracy vs Training Time** --- displays the
    relationship between validation accuracy and training time.

Plots are saved as PNG files in the `ANN_Plots/` directory.

## Requirements

Install the packages used by the experiment and plotting scripts:

``` bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

The exact requirements may vary with the complete notebook and
preprocessing code.

## How to Run

1.  Open the notebook in a Python environment with the required packages
    installed.
2.  Load the dataset and run the preprocessing cells.
3.  Run the ANN experiment code.
4.  Ensure the results are stored in `ANN_computation_results.csv` or
    available as a DataFrame named `results_df`.
5.  Run the plotting cell to generate the five visualizations.

If the results CSV already exists, plots can be generated without
retraining the models.

## Output Files

Expected outputs:

-   `ANN_computation_results.csv` --- recorded metrics for the parameter
    experiments.
-   `ANN_Plots/1_Training_Time.png`
-   `ANN_Plots/2_Validation_Accuracy.png`
-   `ANN_Plots/3_Hidden_Layers_Time.png`
-   `ANN_Plots/4_Activation_Accuracy.png`
-   `ANN_Plots/5_Accuracy_vs_Time.png`

## Interpreting Results

-   Higher validation accuracy generally indicates better classification
    performance on validation data.
-   Lower training time indicates faster training under the recorded
    hardware and software conditions.
-   The accuracy-versus-time plot helps compare model performance and
    computational cost.
-   Results may vary with the dataset, preprocessing, hardware, random
    seed, and training configuration.

Use validation metrics for model selection. If a separate test set is
available, reserve it for final evaluation rather than repeatedly
selecting models based on test-set performance.

## Notes

-   This README is based on the parameter combinations and result-column
    names currently available.
-   Confirm that all 81 experiments completed before assuming the CSV
    contains all combinations.
-   The plotting script expects the column names shown in the Recorded
    Results table.
