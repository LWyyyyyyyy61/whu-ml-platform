from django.shortcuts import render, redirect, HttpResponse
from machinelearn.models import User
from django.contrib.auth import login, authenticate
import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
from .Linear_Regression import training
from .Decision_Tree_R import train
from django.core.files.storage import default_storage
from django.contrib.auth.decorators import login_required
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
    if request.method=='GET':
        return render(request,"home.html")
    else:
        exeq=request.POST.get('exerfqt')
        frq=request.POST.get('learnfqt')
        ct=request.POST.get('cont')

        if (frq and ct and exeq):
                target=request.POST.get('target1')
                uploadfile=request.FILES['exampleInputFile1']
                classop=request.POST.get('qescls1')
                if classop=='cls':
                    file_path = default_storage.save(uploadfile.name, uploadfile)
                    training(file_path, target)
                    image_path = default_storage.url(f'regression_plot_{1000}.png')
                    model_url = default_storage.url(f'linear_model_0.pth')
                    return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
                elif classop=='bak':
                    file_path = default_storage.save(uploadfile.name, uploadfile)
                    training(file_path, target,exeq,ct,frq)
                    image_path = default_storage.url(f'regression_plot_{ct}.png')
                    model_url = default_storage.url(f'linear_model_0.pth')
                    return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        elif ((frq and ct and ~exeq)or(~frq and ct and exeq)or(frq and ~ct and exeq)or(~frq and ~ct and exeq)or(frq and ~ct and ~exeq)or(~frq and ct and ~exeq)):
            return HttpResponse("提交失败请完整填写参数")
        else:
            target=request.POST.get('target')
            uploadfile=request.FILES['exampleInputFile']
            classop=request.POST.get('qescls')    
            if classop=='cls':
                file_path = default_storage.save(uploadfile.name, uploadfile)
                training(file_path, target)
                image_path = default_storage.url(f'regression_plot_{1000}.png')
                model_url = default_storage.url(f'linear_model_0.pth')
                return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
            elif classop=='bak':
                file_path = default_storage.save(uploadfile.name, uploadfile)
                training(file_path, target)
                image_path = default_storage.url(f'regression_plot_{1000}.png')
                model_url = default_storage.url(f'linear_model_0.pth')
                return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})

def upload_file(request):
    
    if request.method == 'POST':
        
        form = UploadFileForm(request.POST, request.FILES)
        classoption=form.cleaned_data['classoption']
        if classoption == 'cls':
            if form.is_valid():
                uploaded_file = request.FILES['file']
                target_column = form.cleaned_data['target_column']
                
                file_path = default_storage.save(uploaded_file.name, uploaded_file)
                training(file_path, target_column)
                image_path = default_storage.url(f'regression_plot_{1000}.png')
                model_url = default_storage.url(f'linear_model_0.pth')
                return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        elif classoption == 'back':
            if form.is_valid():
                uploaded_file = request.FILES['file']
                target_column = form.cleaned_data['target_column']
                file_path = default_storage.save(uploaded_file.name, uploaded_file)
                training(file_path, target_column)
                image_path = default_storage.url(f'regression_plot_{1000}.png')
                model_url = default_storage.url(f'linear_model_0.pth')
                return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        else:
            form = UploadFileForm()
            return render(request, 'upload.html', {'form': form})
    else:
        form = UploadFileForm()
        return render(request, 'upload.html', {'form': form})
    
def userinf(request):
    return render(request,'user.html')
def information(request):
    return render(request,'information.html')