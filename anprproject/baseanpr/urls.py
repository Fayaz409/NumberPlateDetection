from django.urls import path
from . import views2

urlpatterns = [
    path('video_feed/', views2.video_feed, name='video_feed'),
    path('upload_video/', views2.upload_video, name='upload_video'),
    path('process/', views2.process_page, name='process_page'),
    path('', views2.index, name='index')
]

