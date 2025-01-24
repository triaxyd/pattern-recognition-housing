# IMPLEMENTATION OF PERCEPTRON ALGORITHM
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


# Load the data
data = pd.read_csv('housing_scaled.csv')

# Initialize weights and bias
def initialize_weights_bias(n_features):
    weights = np.random.rand(n_features)
    bias = np.random.rand(1)
    return weights, bias

# Define the activation function
def activation_function(x):
    return 1 if x >= 0 else -1

# Define the perceptron function
def perceptron(input, weights, bias):
    return activation_function(np.dot(input, weights) + bias)

# Define the training function
def train_perceptron(X, y, weights, bias, learning_rate, epochs):
    for epoch in range(epochs):
        for i in range(len(X)):
            linear_output = np.dot(X[i], weights) + bias
            prediction = activation_function(linear_output)
            if prediction != y[i]:  # Update weights and bias
                weights += learning_rate * y[i] * X[i]
                bias += learning_rate * y[i]
    return weights, bias

# Define binary target variable
threshold = data['median_house_value'].median()
data['binary_target'] = (data['median_house_value'] > threshold).astype(int) * 2 - 1  # -1, +1

# Features (X) and target (y)
X = data.drop(columns=['median_house_value', 'binary_target']).values
y = data['binary_target'].values

# Parameters
learning_rate = 0.01
epochs = 100

# Cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_list, mae_list = [], []

for train_index, test_index in kf.split(X):
    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize weights and bias
    weights, bias = initialize_weights_bias(X_train.shape[1])
    
    # Train perceptron
    weights, bias = train_perceptron(X_train, y_train, weights, bias, learning_rate, epochs)
    
    # Predictions on test set
    y_pred = np.array([perceptron(x, weights, bias) for x in X_test])
    
    # Calculate errors
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    mse_list.append(mse)
    mae_list.append(mae)

# Results of 10-fold CV
print("Mean MSE (10-fold CV):", np.mean(mse_list))
print("Mean MAE (10-fold CV):", np.mean(mae_list))

# Print value counts of binary target
print(data['binary_target'].value_counts())





