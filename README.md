# Simple Linear Regression

This is a Python implementation of Simple Linear Regression, a basic machine learning algorithm used for predicting a continuous target variable based on a single feature.

## Introduction

Simple Linear Regression is a statistical model that assumes a linear relationship between the feature and the target variable. It fits a line to the data by finding the best slope and intercept values that minimize the sum of squared errors.

This implementation provides a `SimpleLinearRegression` class with methods to train the model, make predictions, and calculate error metrics.

## Features

- Train the model with a feature matrix and target vector.
- Predict target values for new data points.
- Calculate error metrics such as Mean Square Error (MSE) and Mean Absolute Error (MAE).

## Usage

To use this implementation, follow these steps:

1. Import the necessary libraries and the `SimpleLinearRegression` class.
2. Create an instance of the `SimpleLinearRegression` class.
3. Train the model by calling the `train` method with the feature matrix and target vector.
4. Make predictions by calling the `predict` method with new data points.
5. Calculate error metrics using the `error` method.

Example usage:

```python
import numpy as np
from SimpleLinearRegression import SimpleLinearRegression

### Create a feature matrix and target vector
X = np.array([2, 3, 4, 5, 3, 2, 3, 4, 5, 4, 3, 2, 2, 3, 4, 4, 5, 2, 2, 1, 2, 3, 4, 4, 5, 5, 6, 6, 5, 4, 3, 3, 4, 3])
y = np.array([2, 3, 4, 5, 3, 2, 3, 4, 5, 4, 3, 2, 2, 3, 4, 4, 5, 2, 2, 1, 2, 3, 4, 4, 5, 5, 6, 6, 5, 4, 3, 3, 4, 3])

### Instantiate the regressor
regressor = SimpleLinearRegression()

### Train the model
regressor.train(X, y)

### Make predictions
predictions = regressor.predict([6, 7, 8])

### Calculate error metrics
mse = regressor.error(X, y, metric="mean_square_error")
mae = regressor.error(X, y, metric="mean_absolute_error")
For more details on using the SimpleLinearRegression class and its methods, please refer to the code documentation.

Requirements
This implementation requires the following:

Python 3.x
NumPy

# Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

License

Feel free to customize this README file to suit your specific needs and add a