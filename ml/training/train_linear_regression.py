from ml.models.linear_regression_model import load_data, train_model, plot_results

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


data_file_path = 'ml/datasets/linear_data.csv'

# Load the data
X, y = load_data(data_file_path)

# Train the model
model, X_train, y_train, X_test, y_test, y_pred, mse, r2 = train_model(X, y)

# Print the results
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Plot the results
plot_results(X_train, y_train, X_test, y_test, y_pred)