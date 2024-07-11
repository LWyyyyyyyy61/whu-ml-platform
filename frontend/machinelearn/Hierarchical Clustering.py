import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import joblib
from scipy.cluster.hierarchy import dendrogram
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

    return X

def plot_dendrogram(model, **kwargs):
    # 创建连接矩阵
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # 叶节点的计数
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # 绘制树状图
    dendrogram(linkage_matrix, **kwargs)

def train_and_evaluate_model(X, model_name,random_state=65536):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(X)

    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.title(f'{model_name} Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.savefig(f"media/{model_name}_dendrogram.png")
    plt.close()

def training(file_path, random_state=65536):
    model_name = 'Agglomerative'
    model_path = f'media/{model_name}_model.joblib'
    random_state = int(random_state)
    X = preprocess_data_unsupervised(file_path)
    train_and_evaluate_model(X, model_name, random_state=random_state)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(X)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# if __name__ == "__main__":
    # training('F:/datasets/datasets/iris.csv', random_state=65536)
