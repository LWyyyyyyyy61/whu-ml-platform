import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def read_data(file_path, column_names=None):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.data'):
        return pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
    else:
        raise ValueError("Unsupported file format")

def preprocess_data(file_path, target_column, column_names=None):
    df = read_data(file_path, column_names)

    if 'rownames' in df.columns:
        df.drop('rownames', axis=1, inplace=True)

    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)

    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    if df.empty:
        raise ValueError("Dataframe is empty after handling NaN values.")

    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    return X, y, label_encoders

def plot_clusters(X, labels, centroids, title='K-Means Clustering'):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.close()

def train_and_evaluate_model(X, y, num_clusters, model_name):
    model = KMeans(n_clusters=num_clusters, random_state=42)
    labels = model.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    centroids = model.cluster_centers_

    plot_clusters(X, labels, centroids, title=f'{model_name} Clustering')
    print(f"{model_name} Silhouette Score: {silhouette_avg:.2f}")

def main():
    file_path = '/home/asus/iris.csv'  # 修改为你的数据文件路径
    target_column = 'Species'  # 修改为你的目标列名称
    column_names = None  # 如果是 .data 文件，请提供列名称列表
    model_path = 'kmeans_model.joblib'
    num_clusters = 3  # 修改为所需的聚类数量

    X, y, label_encoders = preprocess_data(file_path, target_column, column_names)

    train_and_evaluate_model(X, y, num_clusters, 'K-Means')

    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
