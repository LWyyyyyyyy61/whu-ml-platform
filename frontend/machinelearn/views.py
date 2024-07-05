from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
# Create your views here.

def login(request):
    if request.method == 'POST':
        username = request.POST.get('user')
        password = request.POST.get('pwd')
        # 在此处处理用户名和密码的验证逻辑
        return redirect('/home/')
    return render(request, "login.html")

def register(request):
    if request.method == 'POST':
        username = request.POST['user']
        password = request.POST['pwd']
        mail=request.POST['e-mail']
        phonenumber=request.POST['phone-number']
        if User.objects.filter(username=username).exists():
            # 用户名已存在,提示用户尝试其他用户名
            return render(request, 'register.html', {'error': 'Username already exists. Please try a different one.'})
        user = User.objects.create_user(username=username, password=password,email=mail,first_name=phonenumber)
        user.save()
        login(request, user)
        return redirect('/home/')
    return render(request, "register.html")
def home(request):
    return render(request,"home.html")



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

    plt.figure(figsize=(10, 6))
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

    y_margin = (max(y) - min(y)) * 0.1
    x_margin = (max(X) - min(X)) * 0.1
    plt.ylim(min(y) - y_margin, max(y) + y_margin)
    plt.xlim(min(X) - x_margin, max(X) + x_margin)

    plt.tight_layout()
    plt.savefig(f"media/regression_plot_{iteration}.png")
    plt.close()

def process_and_train_model(file_path, target_column):
    df = pd.read_csv(file_path)

    if 'rownames' in df.columns:
        df.drop('rownames', axis=1, inplace=True)

    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    pca = PCA(n_components=1)
    X = pca.fit_transform(df.drop(target_column, axis=1))

    X = X.flatten()
    y = df[target_column].values

    return X, y, target_column

def custom_linear_regression(X, y, train_ratio=0.8):
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
            theta0 = model.linear.bias.item()
            theta1 = model.linear.weight.item()
            plotLine(theta0, theta1, X_train.numpy(), y_train.numpy(), (i + 1) / 1000, i + 1)

    return model, split_idx

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            target_column = form.cleaned_data['target_column']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)

            X, y, target = process_and_train_model(file_path, target_column)
            model, split_idx = custom_linear_regression(X, y, 0.8)

            torch.save(model.state_dict(), 'media/linear_model.pth')
            model_url = fs.url('linear_model.pth')

            return render(request, 'result.html', {'file_url': fs.url(filename), 'model_url': model_url, 'iteration': 1000})
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})