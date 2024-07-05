import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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

    # 将目标变量转换为整数类别
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    label_encoders[target_column] = le

    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    return X, y, label_encoders

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, label_encoders, target_column):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    target_labels = label_encoders[target_column].classes_

    plot_confusion_matrix(cm, target_labels, title=f'{model_name} Confusion Matrix')
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"{model_name} Confusion Matrix:\n{cm}")

def main():
    file_path = '/home/asus/Thyroid_Diff.csv'  # 修改为你的数据文件路径
    target_column = 'Recurred'  # 修改为你的目标列名称
    column_names = None  # 如果是 .data 文件，请提供列名称列表
    model_path = 'logistic_regression_model.joblib'

    X, y, label_encoders = preprocess_data(file_path, target_column, column_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42)
    train_and_evaluate_model(model, X_train, y_train, X_test, y_test, 'Logistic Regression', label_encoders, target_column)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
