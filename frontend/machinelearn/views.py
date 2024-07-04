from django.shortcuts import render

# Create your views here.

def login(request):
    return render(request,"login.html")

def login_index(request):
    return render(request,"login_index.html")
def home(request):
    return render(request,"home.html")