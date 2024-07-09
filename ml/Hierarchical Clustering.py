import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import joblib
import scipy.cluster.hierarchy as sch


def load_data(file_path):
    """
    Load CSV file into a DataFrame
    """
    data = pd.read_csv(file_path)
    return data


def preprocess_data_unsupervised(file_path):
    """
    Preprocess the data for unsupervised learning
    """
    data = load_data(file_path)
    # Assuming the CSV has no header and we need to consider all columns
    X = data.values
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def plot_dendrogram(X, title='Hierarchical Clustering Dendrogram'):
    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Euclidean distances')
    plt.savefig(f"{title}.png")
    plt.close()


def plot_clusters(X, labels, title='Hierarchical Clustering'):
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = [plt.cm.nipy_spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.close()


def train_and_evaluate_model(X, num_clusters, model_name):
    model = AgglomerativeClustering(n_clusters=num_clusters)
    labels = model.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    print(f"{model_name} Silhouette Score: {silhouette_avg:.2f}")

    plot_clusters(X, labels, title=f'{model_name} Clustering')


def main():
    file_path = 'F:/Google/data2.csv'  # 修改为你的数据文件路径
    model_path = 'hierarchical_model.joblib'
    num_clusters = 2 # 修改为所需的聚类数量

    X = preprocess_data_unsupervised(file_path)

    plot_dendrogram(X)

    train_and_evaluate_model(X, num_clusters, 'Hierarchical')

    model = AgglomerativeClustering(n_clusters=num_clusters)
    model.fit(X)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
