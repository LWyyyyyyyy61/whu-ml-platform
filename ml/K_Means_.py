import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from django.core.files.storage import default_storage

def read_data(file_path, column_names=None):
    local_file_path = default_storage.path(file_path)  # 获取文件的本地路径
    if local_file_path.endswith('.csv'):
        return pd.read_csv(local_file_path)
    elif local_file_path.endswith('.data'):
        return pd.read_csv(local_file_path, sep='\s+', header=None, names=column_names)
    else:
        raise ValueError("Unsupported file format")

def preprocess_data_unsupervised(file_path):
    df = read_data(file_path)  # 转换为DataFrame

    # 删除指定列名（如果有的话）
    if 'rownames' in df.columns:
        df.drop('rownames', axis=1, inplace=True)

    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # 填充缺失值
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)

    # 检查是否还有缺失值
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    if df.empty:
        raise ValueError("Dataframe is empty after handling NaN values.")

    # 标准化数值型特征
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    X = df.values  # 返回预处理后的特征矩阵
    feature_names = df.columns.tolist()  # 获取特征名

    return X, feature_names

def plot_clusters(X, labels, centroids, feature_names, title='K-Means Clustering'):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plot_path = default_storage.save(f"{title}.png", plt)
    plt.close()
    return plot_path

def train_and_evaluate_model(X, num_clusters, feature_names, model_name):
    model = KMeans(n_clusters=num_clusters, random_state=42)
    labels = model.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    centroids = model.cluster_centers_

    plot_path = plot_clusters(X, labels, centroids, feature_names, title=f'{model_name} Clustering')
    print(f"{model_name} Silhouette Score: {silhouette_avg:.2f}")
    return plot_path

def main(file_path,num_clusters):
    model_path = 'media/kmeans_model.joblib'

    X, feature_names = preprocess_data_unsupervised(file_path)
    plot_path = train_and_evaluate_model(X, num_clusters, feature_names, 'K-Means')

    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X)
    joblib.dump(model, default_storage.save(model_path, model))
    print(f"Model saved to {model_path}")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main('/home/asus/datasets/Thyroid_Diff.csv',3)
