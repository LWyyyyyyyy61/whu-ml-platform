from django.shortcuts import render, redirect
from machinelearn.models import User
from django.contrib.auth import login, authenticate
import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
from .Linear_Regression import training
from django.core.files.storage import default_storage

import numpy as np

# Create your views here.

def login(request):
    if request.method == 'GET':
        return render(request, "login.html")
    else:
        username = request.POST.get('user')
        password = request.POST.get('pwd')
    if username and password and User.objects.filter(username=username, password=password).exists():
        return redirect('/home/')
    else:
        return redirect('/login/')
def register(request):
    if request.method == 'POST':
        username = request.POST.get('user')
        password = request.POST.get('pwd')
        email=request.POST.get('e-mail')
        # phonenumber=request.POST.get('phone-number')
        if User.objects.filter(username=username).exists():
            # 用户名已存在,提示用户尝试其他用户名
            return render(request, 'register.html', {'error': 'Username already exists. Please try a different one.'})
        user = User.objects.create(username=username,password=password,email=email,phonenumber="1234567890")
        return redirect('/login/')
    return render(request, "register.html")
def home(request):
    return render(request,"home.html")

def upload_file(request):
     if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            target_column = form.cleaned_data['target_column']
            
            # Save the file using default_storage only once
            file_path = default_storage.save(uploaded_file.name, uploaded_file)

            # Call the training function
            training(file_path, target_column)

            
            # Construct model URL based on default_storage's URL method
            model_url = default_storage.url(f'linear_model.pth')

            return render(request, 'result.html', { 'model_url': model_url})
     else:
        form = UploadFileForm()
     return render(request, 'upload.html', {'form': form})