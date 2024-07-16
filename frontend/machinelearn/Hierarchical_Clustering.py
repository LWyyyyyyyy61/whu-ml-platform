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
        return pd.read_csv(local_file_path)
    elif local_file_path.endswith('.data'):
        return pd.read_csv(local_file_path, sep='\s+', header=None, names=column_names)
    else:
        raise ValueError("Unsupported file format")

def preprocess_data_unsupervised(file_path):
    """
    Preprocesses the unsupervised data by performing the following steps:
    1. Reads the data from the given file path and converts it to a DataFrame.
    2. Removes the 'rownames' column if present.
    3. Encodes categorical columns using LabelEncoder.
    4. Fills missing values with the mode for object columns and the mean for numeric columns.
    5. Drops rows with any remaining missing values.
    6. Standardizes numeric columns using StandardScaler.
    7. Returns the preprocessed feature matrix.

    Args:
        file_path (str): The path to the data file.

    Returns:
        numpy.ndarray: The preprocessed feature matrix.

    Raises:
        ValueError: If the DataFrame is empty after handling NaN values.
    """
    df = read_data(file_path)  # Convert to DataFrame

    # Remove specified column name (if any)
    if 'rownames' in df.columns:
        df.drop('rownames', axis=1, inplace=True)

    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Fill missing values
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)

    # Check if there are any remaining missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    if df.empty:
        raise ValueError("Dataframe is empty after handling NaN values.")

    # Standardize numeric features
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    X = df.values  # Return the preprocessed feature matrix

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
    plt.savefig(f"media/{random_state}_dendrogram.png")
    plt.close()

def training7(file_path, random_state=65536):
    model_name = 'Agglomerative'
    model_path = f'media/{random_state}_model.joblib'
    random_state = int(random_state)
    X = preprocess_data_unsupervised(file_path)
    train_and_evaluate_model(X, model_name, random_state=random_state)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(X)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")