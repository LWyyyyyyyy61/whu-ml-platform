import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from django.core.files.storage import default_storage


def process_and_train_model(file_path, target_column):
    # 获取文件的本地路径
    local_file_path = default_storage.path(file_path)

    # 使用正确的路径读取 CSV 文件
    df = pd.read_csv(local_file_path)

    # 删除不必要的列（如果 'rownames' 列无用，可以删除）
    if 'rownames' in df.columns:
        df.drop('rownames', axis=1, inplace=True)

    # 处理类别变量
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # 特征缩放
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # 使用PCA将特征降维到一维
    pca = PCA(n_components=1)
    X = pca.fit_transform(df.drop(target_column, axis=1))

    # 取得降维后的特征数组
    X = X.flatten()

    y = df[target_column].values

    # 返回 X, y,目标名
    return X, y, target_column

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def plot_predictions(model, X, y, feature_name, iteration):
    X_plot = torch.linspace(X.min(), X.max(), 1000).view(-1, 1)
    y_plot = model(X_plot).detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X_plot.numpy(), y_plot, color='red', label='Model Prediction')

    y_pred = model(X).detach().numpy()
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    plt.text(0.95, 0.95, f'R-squared: {r2:.2f}\nMSE: {mse:.2f}',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.xlabel(feature_name)
    plt.ylabel('Target')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"media/Dec_linear_model_plot_{iteration}.png")
    plt.close()

def train_model(X, y, train_ratio=0.8, num_epochs=1000, lr=0.01):
    # 将数据转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = SimpleLinearModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    return model, X_test, y_test

def train(file_path, target_column, train_ratio, num_epochs, lr):
    X, y,target = process_and_train_model(file_path, target_column)
    train_ratio = float(train_ratio)
    num_epochs = int(num_epochs)
    lr = float(lr)

    feature_array = X.reshape(-1, 1)
    model, X_test, y_test = train_model(feature_array, y, train_ratio, num_epochs, lr)

    # 保存模型
    torch.save(model.state_dict(), f'media/Dec_linear_model_{0}.pth')

    # 加载模型
    loaded_model = SimpleLinearModel(feature_array.shape[1])
    loaded_model.load_state_dict(torch.load(f'media/Dec_linear_model_{0}.pth'))
    loaded_model.eval()

    # 测试加载的模型
    with torch.no_grad():
        test_predictions = loaded_model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        print(f'Test Predictions for Dec_model {0}:', test_predictions)

    # 绘制预测结果
    plot_predictions(loaded_model, torch.tensor(feature_array, dtype=torch.float32), y, feature_name=f"Feature {0}", iteration=10)

# if __name__ == '__main__':
#     train('F:/datasets/datasets/test.csv','education',0.8,10000,0.005)