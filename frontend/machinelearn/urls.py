from django.urls import path
from . import views

urlpatterns = [
   
    path('upload/', views.upload_file, name='upload_file'),
    path('Regressoion/',views.Regress,name='Regress'),
    path('LinearRegression/',views.linearRegress,name='LinearRegress'),
    path('DecisionTree/',views.DecisionTr,name='DesTree')
]