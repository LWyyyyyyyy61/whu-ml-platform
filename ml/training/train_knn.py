from ml.models.knn_model import load_data, train_model, plot_results

data_file_path = 'ml/datasets/dataset.csv'
X, y = load_data(data_file_path)
model, X_train, y_train, X_test, y_test, y_pred, mse, r2 = train_model(X, y)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
plot_results(X_train, y_train, X_test, y_test, y_pred)
