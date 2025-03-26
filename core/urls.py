from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('slr/', views.slr_prediction, name='slr_prediction'),
    path('mlr/', views.mlr_prediction, name='mlr_prediction'),
    path('logistic/', views.logistic_prediction, name='logistic_prediction'),
    path('polynomial/', views.polynomial_prediction, name='polynomial_prediction'),
    path('traffic/', views.traffic_flow_prediction, name='traffic_flow_prediction'),
    path('knn/', views.knn_prediction, name='knn_prediction'),
]

