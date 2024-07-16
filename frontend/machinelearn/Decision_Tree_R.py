import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
import joblib
from django.core.files.storage import default_storage

def process_and_train_model(file_path, target_column):
    # 获取文件的本地路径
    local_file_path = default_storage.path(file_path)

    # 使用正确的路径读取 CSV 文件
    df = pd.read_csv(local_file_path)

    if 'rownames' in df.columns:
        df.drop('rownames', axis=1, inplace=True)

    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    pca = PCA(n_components=1)
    X = pca.fit_transform(df.drop(target_column, axis=1))
    X = X.flatten()

    y = df[target_column].values

    return X, y, target_column

def plot_predictions(model, X, y, train_ratio, iteration):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')

    x_plot = np.linspace(min(X), max(X), 1000).reshape(-1, 1)
    y_plot = model.predict(x_plot)
    plt.plot(x_plot, y_plot, color='red', label='Regression Line')

    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    y_pred = model.predict(X.reshape(-1, 1))
    ss_residual = np.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    mse = np.mean((y_pred - y) ** 2)
    data_utilization = train_ratio * 100

    plt.text(0.95, 0.95, f'R-squared: {r2:.2f}\nMSE: {mse:.2f}\nData Utilization: {data_utilization:.0f}%',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.legend()
    plt.savefig(f"media/Decision_Regression_{iteration}.png")
    plt.close()

def custom_decision_tree_regression(X, y, train_ratio=0.8, max_depth=10):
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx].reshape(-1, 1), X[split_idx:].reshape(-1, 1)
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)

    plot_predictions(model, X_train, y_train, train_ratio, max_depth)

    return model, split_idx

def train(file_path, target_column, train_ratio, max_depth):
    train_ratio =float(train_ratio)
    max_depth = int(max_depth)
    X, y, target = process_and_train_model(file_path, target_column)
    model, split_idx = custom_decision_tree_regression(X, y, train_ratio, max_depth)
   
    model_path = f'media/decision_tree_model_{max_depth}.pkl'
    joblib.dump(model, model_path)

    loaded_model = joblib.load(model_path)

    test_predictions = loaded_model.predict(X[split_idx:].reshape(-1, 1))
    print(f'Test Predictions for model max_depth:', test_predictions)
