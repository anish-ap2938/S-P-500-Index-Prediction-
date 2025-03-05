# S&P 500 Index Prediction

This repository contains the code for predicting the S&P 500 index closing prices for the upcoming eight months (January 2021 to August 2021) using historical data from 2017 to 2020. The project focuses on implementing the gradient descent method from scratch and tuning the sequence length parameter to optimize model performance.

---

## Overview

- **Objective:**  
  Predict future closing prices of the S&P 500 (SPY) by using a sliding window (sequence) of historical closing prices. The sequence length (N) represents the number of past data points used to predict the next price.

- **Method:**  
  - **Gradient Descent Implementation:**  
    The model uses an independently implemented gradient descent method to minimize the prediction error.
  - **Sequence Length Tuning:**  
    Experimentation with different sequence lengths to determine the optimal number of past data points for prediction.
  - **Model Tuning:**  
    Adjusting hyperparameters (such as learning rate) to achieve the best possible forecasting performance.

---

## Data : (https://drive.google.com/file/d/1UH1H8dmYuOcfPRPVDYLI1AIIcwPoqVqW/view)

- **Training Data:**  
  `SPY_dataset.csv` – Contains historical closing prices of the SPY from 2017 to 2020.
  
- **Testing Data:**  
  `SPY_dataset.csv` – Testing portion includes closing prices from January 2021 to August 2021.

- **Data Format:**  
  - **Date:** The date of each observation.
  - **Close:** The closing price of the S&P 500 index on the corresponding date.

Download the dataset using the provided Google Drive link (ensure you have access via your NJIT email if required).

---

## Model and Implementation

### Gradient Descent
- The gradient descent algorithm is implemented from scratch to optimize the model parameters.
- The implementation includes all necessary sections marked with `#YOUR CODE` in the sample code.

### Sequence Length
- The model uses a sliding window approach where the previous N closing prices are used to predict the next one.
- The sequence length (N) is a tunable parameter; the optimal value is determined through experimentation.

### Prediction Equation
- The final prediction equation (`y_pred`) is derived based on the optimized model parameters using the chosen sequence length.

### Graphical Output
- The repository includes code to generate graphs showing the model's prediction results compared to the actual closing prices.

---

## Training Details

- **Loss Function:**  
  Mean Squared Error (MSE) is used to measure the prediction error between the predicted and actual closing prices.

- **Optimization:**  
  The gradient descent method is used to update the model parameters iteratively.

- **Hyperparameter Tuning:**  
  - Sequence length (N) is varied to find the optimal window size.
  - Learning rate and other parameters are tuned for best performance.

- **Output:**  
  - A graph displaying the model predictions versus the actual closing prices.
  - Final prediction results for the eight-month test period.

---

## Code Structure

- **Notebook/Python Script:**  
  - Contains the full implementation including:
    - Data loading and preprocessing.
    - Implementation of the gradient descent algorithm.
    - Model training and evaluation.
    - Hyperparameter tuning (including sequence length experiments).
    - Graph generation for visualizing model predictions.
  
- **Dependencies:**  
  - `numpy`, `pandas` for data processing.
  - `matplotlib` for plotting graphs.
  - Standard Python libraries for file handling and mathematical operations.

---

## Conclusion

This project demonstrates a practical application of gradient descent in time series forecasting. By carefully tuning the sequence length and other hyperparameters, the model aims to accurately predict the S&P 500 index closing prices for the upcoming eight months. All code, experiments, and results are provided in this repository.

---
