from django.shortcuts import HttpResponse, render

# 其他的imports...

def login(request):
    #frontend/login目录下的template找文件，按照注册目录去找的
    return render(request,"login.html")

def  tpl(request):
    
    return render(request,'tpl.html')
# 其他的视图函数...

# Create your views here.

