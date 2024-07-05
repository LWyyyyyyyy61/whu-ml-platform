from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
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