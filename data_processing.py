import pandas as pd
import numpy as np
from mlxtend.preprocessing import minmax_scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# Load the data
data = pd.read_csv('housing.csv')


# Differentiate numerical and categorical features
numerical_features = data.select_dtypes(include=[np.number])
#categorical_features = data.select_dtypes(include=['object'])


# Set the categorical feature 'ocean_proximity' values to numerical values
label_encoder = LabelEncoder()
data['ocean_proximity_encoded'] = label_encoder.fit_transform(data['ocean_proximity'])

#print('Numerical features: ', numerical_features.columns)
#print('Categorical features: ', categorical_features.columns)


# Handle missing values in numerical features
for feature in numerical_features.columns:
    median = numerical_features[feature].median()
    numerical_features[feature].fillna(median, inplace=True)


# Apply Min-Max Scaling
data_scaled_minmax = pd.DataFrame(
    minmax_scaling(numerical_features, columns=numerical_features.columns),
    columns=numerical_features.columns
)

# Apply min-max scaling to the encoded ocean proximity
data['ocean_proximity_scaled'] = minmax_scaling(data[['ocean_proximity_encoded']], columns=['ocean_proximity_encoded'])

# Combine numerical features with the scaled ocean proximity
data_scaled = pd.concat([data_scaled_minmax, data[['ocean_proximity_scaled']]], axis=1)


# Save the scaled data to a new CSV file
data_scaled.to_csv('housing_scaled.csv', index=False)



"""
# For each feature, plot the histogram of frequencies of the values
for feature in data_scaled.columns:
    sns.displot(data_scaled[feature], kde=False)
    plt.title('Histogram of ' + feature)
    plt.show()

# Scatter plot for longitude vs latitude
sns.scatterplot(x=data_scaled['longitude'], y=data_scaled['latitude'], hue=data_scaled['median_house_value'])
plt.title("Longitude - Latitude -> Median house value")
plt.show()
"""

# Display all the features in the same plot
plt.figure(figsize=(10, 6))
for feature in data_scaled.columns:
    plt.scatter(data_scaled[feature], data_scaled['median_house_value'], label=feature, alpha=0.5)
plt.xlabel('Feature values')
plt.ylabel('Median house value')
plt.legend()
plt.show()







