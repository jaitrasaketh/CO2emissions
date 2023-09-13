import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv("Fuel.csv")

# Select the features and target variable
features = ['ENGINESIZE', 'FUELCONSUMPTION_COMB']
target = 'CO2EMISSIONS'

# Split the data into training and testing sets
train_ratio = 0.8
train_size = int(train_ratio * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Extract feature matrices and target vectors and convert them to float64
X_train = train_data[features].values.astype(np.float64)
X_test = test_data[features].values.astype(np.float64)
y_train = train_data[target].values.astype(np.float64)
y_test = test_data[target].values.astype(np.float64)

# Add a column of ones to the feature matrices for the bias term
X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

# Initialize custom model parameters
np.random.seed(0)
weights = np.random.rand(X_train.shape[1]).astype(np.float64)
alpha = 0.001  # Learning rate
iterations = 1000

# Define a function to calculate predictions
def predict(X, weights):
    return np.dot(X, weights)

# Training the custom model
for _ in range(iterations):
    # Calculate predicted CO2 emissions
    predictions = predict(X_train, weights)
    
    # Calculate the error
    error = predictions - y_train
    
    # Compute the gradient
    gradient = np.dot(X_train.T, error) / len(X_train)
    
    # Update model parameters
    weights -= alpha * gradient

# Make predictions on the test set
test_predictions_custom = predict(X_test, weights)

# Evaluate the custom model
mse_custom = np.mean((test_predictions_custom - y_test) ** 2)
print(f"Mean Squared Error (Custom Model): {mse_custom}")

# Create and train the scikit-learn Linear Regression model
model_sklearn = LinearRegression()
model_sklearn.fit(X_train[:, 1:], y_train)

# Make predictions using the scikit-learn model
test_predictions_sklearn = model_sklearn.predict(X_test[:, 1:])

# Evaluate the scikit-learn model
mse_sklearn = np.mean((test_predictions_sklearn - y_test) ** 2)
print(f"Mean Squared Error (Scikit-Learn Model): {mse_sklearn}")

# Calculate R-squared for both models
r2_custom = r2_score(y_test, test_predictions_custom)
r2_sklearn = r2_score(y_test, test_predictions_sklearn)

print(f"R-squared for Custom Model: {r2_custom}")
print(f"R-squared for Scikit-Learn Model: {r2_sklearn}")

# Visualization (optional)
plt.scatter(y_test, test_predictions_custom, label='Custom Model', alpha=0.5)
plt.scatter(y_test, test_predictions_sklearn, label='Scikit-Learn Model', alpha=0.5)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs. Predicted CO2 Emissions")
plt.legend()
plt.show()

