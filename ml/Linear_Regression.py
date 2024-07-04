import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

# 全局变量
global_val = 2

def process_and_train_model(file_path, target_column):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 删除不必要的列（如果 rownames 列无用，可以删除）
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

    # 特征和目标变量分离
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    return X, y

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def plotLine(theta0, theta1, X, y, train_ratio, iteration):
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100

    xplot = np.linspace(min_x, max_x, 1000)
    yplot = theta0 + theta1 * xplot

    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(xplot, yplot, color='#58b970', label='Regression Line')
    plt.scatter(X, y)

    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - (theta0 + theta1 * X)) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    mse = np.mean((theta0 + theta1 * X - y) ** 2)
    data_utilization = train_ratio * 100

    plt.text(0.95, 0.95, f'R-squared: {r2:.2f}\nMSE: {mse:.2f}\nData Utilization: {data_utilization:.0f}%',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # 自动设置纵坐标和横坐标范围
    y_margin = (max(y) - min(y)) * 0.1
    x_margin = (max(X) - min(X)) * 0.1
    plt.ylim(min(y) - y_margin, max(y) + y_margin)
    plt.xlim(min(X) - x_margin, max(X) + x_margin)

    plt.tight_layout()  # 自动调整子图参数以适应图像
    plt.savefig(f"regression_plot_{iteration}.png")
    plt.close()

def custom_linear_regression(X, y, train_ratio=0.8):
    # 将数据转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    for i in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            # 获取当前的模型参数
            theta0 = model.linear.bias.item()
            theta1 = model.linear.weight.item()
            plotLine(theta0, theta1, X_train.numpy(), y_train.numpy(), (i + 1) / 1000, i + 1)

    return model, split_idx

# 使用示例
X, y = process_and_train_model('/home/asus/index.csv', 'money')

for j in range(global_val):
    feature_array = X[:, j]
    model, split_idx = custom_linear_regression(feature_array, y, 0.8)

    # 保存模型
    torch.save(model.state_dict(), f'linear_model_{j}.pth')

    # 加载模型
    loaded_model = SimpleLinearModel()
    loaded_model.load_state_dict(torch.load(f'linear_model_{j}.pth'))
    loaded_model.eval()

    # 测试加载的模型
    with torch.no_grad():
        test_predictions = loaded_model(torch.tensor(feature_array[split_idx:], dtype=torch.float32).view(-1, 1)).numpy()
        print(f'Test Predictions for model {j}:', test_predictions)
