import pandas as pd
import numpy as np
from spicy import stats
from mlxtend.preprocessing import minmax_scaling
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('housing.csv')


# Differentiate numerical and categorical features
numerical_features = data.select_dtypes(include=[np.number])
categorical_features = data.select_dtypes(include=['object'])

print('Numerical features: ', numerical_features.columns)
print('Categorical features: ', categorical_features.columns)


# Handle missing values in numerical features
data[numerical_features.columns] = data[numerical_features.columns].fillna(data[numerical_features.columns].mean())


# Apply Min-Max Scaling
data_scaled_minmax = pd.DataFrame(
    minmax_scaling(numerical_features, columns=numerical_features.columns),
    columns=numerical_features.columns
)

# Apply One-Hot Encoding to categorical features
categorical_data_encoded = pd.get_dummies(categorical_features)

# Combine scaled numerical features and encoded categorical features
data_scaled = pd.concat([data_scaled_minmax, categorical_data_encoded], axis=1)

# Check the scaled data
print(data_scaled.head())











