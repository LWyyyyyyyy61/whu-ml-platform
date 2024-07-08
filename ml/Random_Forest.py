import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


def preprocess_data(file_path, target_column):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 删除不必要的列（如果 rownames 列无用，可以删除）
    if 'rownames' in df.columns:
        df.drop('rownames', axis=1, inplace=True)

    # 处理缺失值（填补数值型缺失值为均值，类别型缺失值为众数）
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)

    # 编码类别变量
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # 特征和目标变量分离
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 独热编码
    X = pd.get_dummies(X, drop_first=True)

    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders, scaler

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()

def train_random_forest_classifier(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path)

def test_model(model, X_test, y_test, label_encoders, target_column):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 绘制混淆矩阵热图
    target_labels = list(label_encoders[target_column].classes_)
    plot_confusion_matrix(cm, classes=target_labels, title='RF_Confusion Matrix')

    print(f"Accuracy: {accuracy:.2f}")
    print("Random_Forest_Confusion Matrix:")
    print(cm)

# 使用示例
file_path = 'F:/datasets/datasets/caesarian.csv'
target_column = 'Caesarian'
model_path = 'random_forest_model.joblib'

X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_data(file_path, target_column)

 # 使用所有特征进行训练
model = train_random_forest_classifier(X_train, y_train)
save_model(model, model_path)
loaded_model = load_model(model_path)
test_model(loaded_model, X_test, y_test, label_encoders, target_column)

   
