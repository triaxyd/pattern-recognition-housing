# NON LINEAR REGRESSION MULTILAYERED NEURAL NETWORK  

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

# Load the data
data = pd.read_csv('housing_scaled.csv')

# Features (X) and target (y)
X = data.drop(columns=['median_house_value']).values
y = data['median_house_value'].values

# Parameters
learning_rate = 0.01
epochs = 100
hidden_layer_sizes = (100, 100)

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
#kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

mse_train_list, mse_test_list = [], []
mae_train_list, mae_test_list = [], []


for train_index, test_index in kf.split(X):
    # Proceed with the rest of the code
    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize the MLP regressor
    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate,
        max_iter=epochs,
        early_stopping=True,
        random_state=42
    )
    
    # Train the MLP regressor
    mlp_regressor.fit(X_train, y_train)
    
    # Predictions on training set
    y_train_pred = mlp_regressor.predict(X_train)
    
    # Predictions on test set
    y_test_pred = mlp_regressor.predict(X_test)
    
    # Calculate errors
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    mse_train_list.append(mse_train)
    mae_train_list.append(mae_train)
    
    mse_test_list.append(mse_test)
    mae_test_list.append(mae_test)

# Results of 10-fold CV
print("Mean MSE (10-fold CV) - Training set:", np.mean(mse_train_list))
print("Mean MAE (10-fold CV) - Training set:", np.mean(mae_train_list))
print("Mean MSE (10-fold CV) - Testing set:", np.mean(mse_test_list))
print("Mean MAE (10-fold CV) - Testing set:", np.mean(mae_test_list))