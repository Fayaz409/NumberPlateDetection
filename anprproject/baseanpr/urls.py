from django.urls import path
from . import views

urlpatterns = [
    path('video_feed/', views.video_feed, name='video_feed'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('process/', views.process_page, name='process_page'),
    path('', views.index, name='index')
]

