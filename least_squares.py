import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
data = pd.read_csv('housing_scaled.csv')

# Features (X) and target (y)
X = data.drop(columns=['median_house_value']).values  # Independent variables
y = data['median_house_value'].values  # Dependent variable

# Add bias to X 
X = np.hstack((np.ones((X.shape[0], 1)), X))  

# 10-fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_train_list, mse_test_list = [], []
mae_train_list, mae_test_list = [], []

for train_index, test_index in kf.split(X):
    # Test and train sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Calculate the weights 
    # w = (X^T * X)^-1 * X^T * y
    X_transpose = X_train.T
    w = np.linalg.inv(X_transpose @ X_train) @ X_transpose @ y_train

    # Predictions for the training set = X * w
    y_train_pred = X_train @ w

    # Predictions for the testing set = X * w
    y_test_pred = X_test @ w

    # Calculate errors
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)

    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Append to lists
    mse_train_list.append(mse_train)
    mae_train_list.append(mae_train)
    mse_test_list.append(mse_test)
    mae_test_list.append(mae_test)

# Final results
print("Mean MSE (10-fold CV) - Training set:", np.mean(mse_train_list))
print("Mean MAE (10-fold CV) - Training set:", np.mean(mae_train_list))
print("Mean MSE (10-fold CV) - Testing set:", np.mean(mse_test_list))
print("Mean MAE (10-fold CV) - Testing set:", np.mean(mae_test_list))