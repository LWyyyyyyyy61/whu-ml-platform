import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns


def preprocess_data(file_path, target_column):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 删除不必要的列（如果 rownames 列无用，可以删除）
    if 'rownames' in df.columns:
        df.drop('rownames', axis=1, inplace=True)

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
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # 如果是分类任务，标签需要是 long 类型
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_mlp_classifier(X_train, y_train, input_size, hidden_size, num_classes, num_epochs=100, learning_rate=0.001):
    model = SimpleMLP(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def evaluate_model(model, X_test, y_test, iteration):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        cm = confusion_matrix(y_test.numpy(), predicted.numpy())
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Confusion Matrix:\n{cm}')

        # 保存混淆矩阵图片
        save_confusion_matrix_plot(cm, iteration)

        return accuracy, cm

def save_confusion_matrix_plot(cm, iteration):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for iteration {iteration}')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{iteration}.png")
    plt.close()

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')

def load_model(filepath, input_size, hidden_size, num_classes):
    model = SimpleMLP(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f'Model loaded from {filepath}')
    return model

# 使用示例
file_path = '/home/asus/Thyroid_Diff.csv'
target_column = 'Recurred'
X_train, X_test, y_train, y_test = preprocess_data(file_path, target_column)

input_size = X_train.shape[1]
hidden_size = 128
num_classes = len(np.unique(y_train))

# 训练模型
model = train_mlp_classifier(X_train, y_train, input_size, hidden_size, num_classes, num_epochs=100)

# 评估模型
accuracy, cm = evaluate_model(model, X_test, y_test, iteration=1)

# 保存模型
model_path = 'mlp_model.pth'
save_model(model, model_path)

# 加载模型
loaded_model = load_model(model_path, input_size, hidden_size, num_classes)

# 测试加载的模型
# 假设 X_test 和 y_test 是新的测试数据
# 需要将测试数据转换为 PyTorch 张量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

evaluate_model(loaded_model, X_test_tensor, y_test_tensor, iteration=2)
