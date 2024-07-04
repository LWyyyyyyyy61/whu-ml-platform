from django.shortcuts import render,redirect

# Create your views here.

def login(request):
    if request.method == 'POST':
        username = request.POST.get('user')
        password = request.POST.get('pwd')
        # 在此处处理用户名和密码的验证逻辑
        return redirect('/home/')
    return render(request, "login.html")

# def login_index(request):
#     return render(request,"login_index.html")
def home(request):
    return render(request,"home.html")