from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from django.contrib import auth
import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
from .Linear_Regression import training
from .Decision_Tree_R import train
from .Decision_Tree import training1
from .Logical_Regression import training2
from .Random_Forest import training3
from .SVMClassifier import training4
from .MLPClassifier import training5
from .K_Means_ import training6
from .Hierarchical_Clustering import training7
from .DBSCAN import training8
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
        if username and password:
            # 进行身份验证
            
            user = authenticate(username=username, password=password)
            if user:
                # 读者验证成功，执行逻辑或重定向到其他页面
                auth.login(request,user)
                return redirect('home/')
            else:
               return render(request, 'login.html', {'error': 'Invalid login credentials.'})
        else:
            return HttpResponse('请提供用户名和密码')    
def register(request):
    if request.method == 'POST':
        username = request.POST.get('user')
        password = request.POST.get('pwd')
        email=request.POST.get('e-mail')
        # phonenumber=request.POST.get('phone-number')
        if User.objects.filter(username=username).exists():
            # 用户名已存在,提示用户尝试其他用户名
            return render(request, 'register.html', {'error': 'Username already exists. Please try a different one.'})
        user = User.objects.create_user(username=username,password=password,email=email)
        user.save()
        return redirect('/login/')
    return render(request, "register.html")

@login_required
def home(request):
        return render(request,"home.html")
@login_required
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
@login_required   
def userinf(request):
    return render(request,'user.html')
@login_required
def information(request):
    return render(request,'information.html')
def Regress(request):
    return render(request,'Regression.html')
def Classification(request):
    return render(request,'Classification.html')
def Clustering(request):
    return render(request,'Clustering.html')
def Modelintro(request):
    return render(request,'Modelintro.html')
def linearRegress(request):
    if request.method=='GET':
        return render(request,'Linear_Regression.html')
    else:
        exeq=request.POST.get('exerfqt')
        ct=request.POST.get('cont')
        frq=request.POST.get('learnfqt')
        if (frq and ct and exeq):
            target=request.POST.get('regretar1')
            uploadfile=request.FILES['regresfile1']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training(file_path, target,exeq,ct,frq)
            image_path = default_storage.url(f'regression_plot_{ct}.png')
            model_url = default_storage.url(f'linear_model_{ct}.pth')
            return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        elif(not ct and not exeq and not frq):
            target=request.POST.get('regretar')
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training(file_path, target,0.8,1000,0.005)
            image_path = default_storage.url(f'regression_plot_{1000}.png')
            model_url = default_storage.url(f'linear_model_{1000}.pth')
            return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        else:
            return HttpResponse("提交失败请完整填写参数")
def DecisionTr(request):
    if request.method=='GET':
        return render(request,'Decision_Tree.html')
    else:
        exeq=request.POST.get('exerfqt')
        ct=request.POST.get('cont')
        if (ct and exeq):
                target=request.POST.get('regretar1')
                uploadfile=request.FILES['regresfile1']
                file_path = default_storage.save(uploadfile.name, uploadfile)
                train(file_path, target,exeq,ct)
                image_path = default_storage.url(f'Decision_Regression_{ct}.png')
                model_url = default_storage.url(f'Decision_Regression_{ct}.pkl')
                return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        elif(not ct and not exeq):
            target=request.POST.get('regretar')
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            train(file_path, target,0.8,10)
            image_path = default_storage.url(f'Decision_Regression_{10}.png')
            model_url = default_storage.url(f'Decision_Regression_{10}.pkl')
            return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        else:
            return HttpResponse("提交失败请完整填写参数")
def Detrcfn(request):
    if request.method=='GET':
        return render(request,'classfy/DecisionTreeClassification.html')
    else:
        exeq=request.POST.get('exerfqt')
        ct=request.POST.get('cont')
        if (ct and exeq):
            target=request.POST.get('regretar1')
            uploadfile=request.FILES['regresfile1']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training1(file_path, target,exeq,ct)
            image_path = default_storage.url(f'confusion_matrix_{ct}.png')
            image_path2= default_storage.url(f'decision_tree_plot_{ct}.png')
            model_url = default_storage.url(f'decision_tree_model_{ct}.joblib')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        elif(not ct and not exeq):
            target=request.POST.get('regretar')
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training1(file_path, target,0.8,65536)
            image_path = default_storage.url(f'confusion_matrix_{65536}.png')
            image_path2= default_storage.url(f'decision_tree_plot_{65536}.png')
            model_url = default_storage.url(f'decision_tree_model_{65536}.joblib')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        else:
            return HttpResponse("提交失败请完整填写参数")    
def LoReg(request):
    if request.method=='GET':
        return render(request,'classfy/loginRegress.html')
    else:
        exeq=request.POST.get('exerfqt')
        ct=request.POST.get('cont')
        if (ct and exeq):
            target=request.POST.get('regretar1')
            uploadfile=request.FILES['regresfile1']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training2(file_path, target,exeq,ct)
            image_path = default_storage.url(f'logistic_regression_confusion_matrix_{ct}.png')
            image_path2= default_storage.url(f'logistic_regression_plot_{ct}.png')
            model_url = default_storage.url(f'logistic_regression_model_{ct}.joblib')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        elif(not ct and not exeq):
            target=request.POST.get('regretar')
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training2(file_path, target,0.8,65536)
            image_path = default_storage.url(f'logistic_regression_confusion_matrix_{65536}.png')
            image_path2= default_storage.url(f'logistic_regression_plot_{65536}.png')
            model_url = default_storage.url(f'logistic_regression_model_{65536}.joblib')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        else:
            return HttpResponse("提交失败请完整填写参数")
def RF(request):
    if request.method=='GET':
        return render(request,'classfy/RF.html')
    else:
        exeq=request.POST.get('exerfqt')
        ct=request.POST.get('cont')
        if (ct and exeq):
            target=request.POST.get('regretar1')
            uploadfile=request.FILES['regresfile1']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training3(file_path, target,exeq,ct)
            image_path = default_storage.url(f'random_forest_plot_{ct}.png')
            image_path2= default_storage.url(f'random_forest_confusion_matrix_{ct}.png')
            model_url = default_storage.url(f'random_forest_model_{ct}.joblib')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        elif(not ct and not exeq):
            target=request.POST.get('regretar')
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training3(file_path, target,0.8,65536)
            image_path = default_storage.url(f'random_forest_plot_{65536}.png')
            image_path2= default_storage.url(f'random_forest_confusion_matrix_{65536}.png')
            model_url = default_storage.url(f'random_forest_model_{65536}.joblib')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        else:
            return HttpResponse("提交失败请完整填写参数")
def SVM(request):
    if request.method=='GET':
        return render(request,'classfy/SVM.html')
    else:
        exeq=request.POST.get('exerfqt')
        ct=request.POST.get('cont')
        if (ct and exeq):
            target=request.POST.get('regretar1')
            uploadfile=request.FILES['regresfile1']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training4(file_path, target,exeq,ct)
            image_path = default_storage.url(f'svm_plot_{ct}.png')
            image_path2= default_storage.url(f'svm_confusion_matrix_{ct}.png')
            model_url = default_storage.url(f'svm_model_{ct}.joblib')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        elif(not ct and not exeq):
            target=request.POST.get('regretar')
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training4(file_path, target,0.8,65536)
            image_path = default_storage.url(f'svm_plot_{65536}.png')
            image_path2= default_storage.url(f'svm_confusion_matrix_{65536}.png')
            model_url = default_storage.url(f'svm_model_{65536}.joblib')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        else:
            return HttpResponse("提交失败请完整填写参数")
def MLP(request):
    if request.method=='GET':
        return render(request,'classfy/MLP.html')
    else:
        exeq=request.POST.get('exerfqt')
        ct=request.POST.get('cont')
        sj=request.POST.get('sj')
        learcnt=request.POST.get('learcnt')
        learnfqt=request.POST.get('learnfqt')
        if(ct and exeq and sj and learcnt and learnfqt):
            target=request.POST.get('regretar1')
            uploadfile=request.FILES['regresfile1']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training5(file_path, target,exeq,ct,sj,learcnt,learnfqt)
            image_path = default_storage.url(f'mlp_plot_{ct}.png')
            image_path2= default_storage.url(f'mlp_confusion_matrix_{ct}.png')
            model_url = default_storage.url(f'mlp_model_{ct}.pth')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        elif(not ct and not exeq and not sj and not learcnt and not learnfqt):
            target=request.POST.get('regretar')
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training5(file_path, target,0.8,65536,10,1000,0.001)
            image_path = default_storage.url(f'mlp_plot_{65536}.png')
            image_path2= default_storage.url(f'mlp_confusion_matrix_{65536}.png')
            model_url = default_storage.url(f'mlp_model_{65536}.pth')
            return render(request, 'result2.html', {'model_url': model_url, 'image_path': image_path,'image_path2':image_path2})
        else:
            return HttpResponse("提交失败请完整填写参数")
def Kmeans(request):
    if request.method=='GET':
        return render(request,'cluster/Kmeans.html')
    else:
        exeq=request.POST.get('exerfqt')
        ct=request.POST.get('cont')
        if (exeq and ct):
            uploadfile=request.FILES['regresfile1']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training6(file_path,exeq,ct)
            image_path = default_storage.url(f'{ct}.png')
            model_url = default_storage.url(f'Kmeans_model_{ct}.joblib')
            return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        elif(not exeq and not ct):
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training6(file_path,3,65536)
            image_path = default_storage.url(f'{65536}.png')
            model_url = default_storage.url(f'Kmeans_model_{65536}.joblib')
            return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        else:
            return HttpResponse("提交失败请完整填写参数")
def Hierarchical(request):
    if request.method=='GET':
        return render(request,'cluster/Hierarchical.html')
    else:
        ct=request.POST.get('cont')
        if (ct):
            uploadfile=request.FILES['regresfile1']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training7(file_path,ct)
            image_path = default_storage.url(f'{ct}_dendrogram.png')
            model_url = default_storage.url(f'{ct}_model.joblib')
            return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        else:
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training7(file_path,65536)
            image_path = default_storage.url(f'{65536}_dendrogram.png')
            model_url = default_storage.url(f'{65536}_model.joblib')
            return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
def DBSCAN(request):
    if request.method=='GET':
        return render(request,'cluster/DBSCAN.html')
    else:
        exeq=request.POST.get('exerfqt')
        ct=request.POST.get('cont')
        if (exeq and ct):
            uploadfile=request.FILES['regresfile1']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training8(file_path,exeq,ct)
            image_path = default_storage.url(f'{ct}.png')
            model_url = default_storage.url(f'Kmeans_model_{ct}.joblib')
            return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        elif(not exeq and not ct):
            uploadfile=request.FILES['regresfile']
            file_path = default_storage.save(uploadfile.name, uploadfile)
            training8(file_path,1.0,5)
            image_path = default_storage.url(f'{ct}.png')
            model_url = default_storage.url(f'dbscan_model_{ct}.joblib')
            return render(request, 'result.html', {'model_url': model_url, 'image_path': image_path})
        else:
            return HttpResponse("提交失败请完整填写参数")
       