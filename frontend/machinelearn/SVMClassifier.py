import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from django.core.files.storage import default_storage

def preprocess_data(file_path, target_column, train_ratio=0.8, random_state=65536):
    # 获取文件的本地路径
    local_file_path = default_storage.path(file_path)

    # 使用正确的路径读取 CSV 文件
    df = pd.read_csv(local_file_path)

    # 处理缺失值（填补数值型缺失值为均值，类别型缺失值为众数）
    for column in df.columns:
        if df[column].dtype == 'object':
            df.fillna({column: df[column].mode()[0]}, inplace=True)
        else:
            df.fillna({column: df[column].mean()}, inplace=True)

    # 编码类别变量
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # 特征和目标变量分离
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values

    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=(1 - train_ratio), random_state=random_state)

    return X_train, X_test, y_train, y_test

def plot_svm_results(model, X_test, y_test, random_state,feature_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label='True labels')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', alpha=0.5, label='Predicted labels')

    plt.text(0.95, 0.95, f'Accuracy: {accuracy:.2f}',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.xlabel(f'{feature_name} (Test Data)')
    plt.ylabel('Target')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"media/svm_plot_{random_state}.png")
    plt.close()

    # 显示混淆矩阵并添加数字
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {feature_name}')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(y_test), rotation=45)
    plt.yticks(tick_marks, np.unique(y_test))

    # 添加数字
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"media/svm_confusion_matrix_{random_state}.png")
    plt.close()

def train_svm_classifier(X_train, y_train, random_state=65536):
    model = SVC(kernel='linear', random_state=random_state)  # 使用线性核的 SVM
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f'Model saved to {filepath}')

def load_model(filepath):
    model = joblib.load(filepath)
    print(f'Model loaded from {filepath}')
    return model

def training4(file_path, target_column, train_ratio, random_state):
    # 使用示例
    train_ratio = float(train_ratio)
    random_state = int(random_state)

    X_train, X_test, y_train, y_test = preprocess_data(file_path, target_column, train_ratio, random_state)

    # 使用所有特征进行训练
    model = train_svm_classifier(X_train, y_train, random_state)
    plot_svm_results(model, X_test, y_test, random_state,f"Feature {target_column}")

    # 保存模型
    model_path = f'media/svm_model_{random_state}.joblib'
    save_model(model, model_path)

