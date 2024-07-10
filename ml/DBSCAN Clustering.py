import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import joblib
from django.core.files.storage import default_storage
def read_data(file_path, column_names=None):
    # 获取文件的本地路径
    local_file_path = default_storage.path(file_path)
    if local_file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif local_file_path.endswith('.data'):
        return pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
    else:
        raise ValueError("Unsupported file format")
def preprocess_data_unsupervised(file_path):
    df =read_data(file_path)  # 转换为DataFrame

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

    return X

def plot_clusters(X, labels, title='DBSCAN Clustering'):
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black color for noise
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title(title)
    plt.savefig(f"media/{title}.png")
    plt.close()


def train_and_evaluate_model(X, eps, min_samples, model_name):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    if len(set(labels)) > 1:  # Check if more than one cluster was found
        silhouette_avg = silhouette_score(X, labels)
        print(f"{model_name} Silhouette Score: {silhouette_avg:.2f}")
    else:
        print(f"{model_name} could not find enough clusters for Silhouette Score calculation.")

    plot_clusters(X, labels, title=f'{model_name} Clustering')


def training(file_path,eps, min_samples):
    model_path = '../media/dbscan_model.joblib'
    eps=float(eps)
    min_samples=int(min_samples)

    X = preprocess_data_unsupervised(file_path)

    train_and_evaluate_model(X, eps, min_samples, 'DBSCAN')

    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


# if __name__ == "__main__":
#     training('F:/datasets/datasets/iris.csv',1.0,5)

