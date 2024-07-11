from django.urls import path
from . import views

urlpatterns = [
   
    path('upload/', views.upload_file, name='upload_file'),
    path('Regressoion/',views.Regress,name='Regress'),
    path('LinearRegression/',views.linearRegress,name='LinearRegress'),
    path('DecisionTree/',views.DecisionTr,name='DesTree'),
    path('Classificaton/',views.Classification,name='Classification'),
    path('Clustering/',views.Clustering,name='Clustering'),
    path('Modelintro/',views.Modelintro,name='Modelintro'),
    path('DeTrCn/',views.Detrcfn,name='Detrcn'),
    path('loginRe/',views.LoReg,name='LoReg'),
    path('RF/',views.RF,name='RF'),
    path('SVM/',views.SVM,name='SVM'),
    path('MLP/',views.MLP,name='MLP'),
    path('kmeans/',views.Kmeans,name='Kmeans'),
    path('Hierarchical/',views.Hierarchical,name='Hierarchical'),
    path('DBSCAN/',views.DBSCAN,name='DBSCAN'),

]