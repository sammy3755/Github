import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.X = None  # Feature matrix
        self.y = None  # Target vector
        self.slope_ = None  # Slope of the regression line
        self.intercept_ = None  # Intercept of the regression line
    
    def train(self, X, y):
        if len(X) != len(y):
            print("The lengths of the feature matrix and the target vector must be equal")
            return 
            
        self.X = np.array(X)  # Convert feature matrix to NumPy array
        self.y = np.array(y)  # Convert target vector to NumPy array
        
        # Calculate necessary summations
        sum_xy = (self.X * self.y).sum()
        sum_x = self.X.sum()
        sum_y = self.y.sum()
        sum_xSq = (self.X**2).sum()
        n = len(self.X)
        
        # Calculate slope and intercept of the regression line
        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_xSq - sum_x**2
        self.slope_ = numerator / denominator
        self.intercept_ = self.y.mean() - self.slope_ * self.X.mean()
    
    def predict(self, unknown):
        self.y_pred = []  # Predicted target values
        
        # Iterate over each element in unknown and calculate predicted target value
        for x in unknown:
            self.y_pred.append(self.intercept_ + self.slope_ * x)
        
        self.y_pred = np.array(self.y_pred)  # Convert predictions to NumPy array
        return self.y_pred
    
    def error(self, X, y, metric="mean_square_error"):
        self.X = np.array(X)  # Convert feature matrix to NumPy array
        self.y = np.array(y)  # Convert target vector to NumPy array
        self.y_test_pred = self.predict(self.X)  # Predict target values for X
        
        if metric == "mean_square_error":
            self.MSE = ((self.y - self.y_test_pred)**2).mean()  # Calculate mean square error
            return self.MSE
        elif metric == "mean_absolute_error":
            self.MAE = np.abs(self.y - self.y_test_pred).mean()  # Calculate mean absolute error
            return self.MAE

# Dataset
X = [2, 3, 4, 5, 3, 2, 3, 4, 5, 4, 3, 2, 2, 3, 4, 4, 5, 2, 2, 1, 2, 3, 4, 4, 5, 5, 6, 6, 5, 4, 3, 3, 4, 3]
y = [2, 3, 4, 5, 3, 2, 3, 4, 5, 4, 3, 2, 2, 3, 4, 4, 5, 2, 2, 1, 2, 3, 4, 4, 5, 5, 6, 6, 5, 4, 3, 3, 4, 3]

# Split dataset into training and prediction sets (80%:20%)
split_index = int(len(X) * 0.8)
X_train = X[:split_index]
y_train = y[:split_index]
X_pred = X[split_index:]
y_pred = y[split_index:]

# Instantiate the regressor
regressor = SimpleLinearRegression()

# Train the model with the training set
regressor.train(X_train, y_train)

# Predict target values for the remaining data
predictions = regressor.predict(X_pred)
print("Predictions:", predictions)

# Calculate errors
mse = regressor.error(X_pred, y_pred, metric="mean_square_error")
mae = regressor.error(X_pred, y_pred, metric="mean_absolute_error")
print("Mean Square Error:", mse)
print("Mean Absolute Error:", mae)
