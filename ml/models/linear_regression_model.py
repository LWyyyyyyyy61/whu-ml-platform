import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 0].values.reshape(-1, 1)
    y = data.iloc[:, 0].values.reshape(-1, 1)
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, X_train, y_train, X_test, y_test, y_pred, mse, r2\
    
def plot_results(X_train, y_train, X_test, y_test, y_pred):
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.scatter(X_test, y_test, color='green', label="Test Data")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()