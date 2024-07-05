from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
from .Linear_Regression import training
from django.core.files.storage import default_storage
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

def upload_file(request):
     if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            target_column = form.cleaned_data['target_column']
            
            # Save the file using default_storage only once
            file_path = default_storage.save(uploaded_file.name, uploaded_file)

            # Call the training function
          
            
            # Construct model URL based on default_storage's URL method
            model_url = default_storage.url(f'linear_model.pth')

            return render(request, 'result.html', { 'model_url': model_url})
     else:
        form = UploadFileForm()
     return render(request, 'upload.html', {'form': form})