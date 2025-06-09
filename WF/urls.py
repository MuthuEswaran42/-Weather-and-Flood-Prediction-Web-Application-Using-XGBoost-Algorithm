from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_views, name='home'),  # <--- default homepage
    path('home/', views.home_views, name='home_alt'),
    path('about/', views.about, name='about'),     # Add missing comma here
    path('upload_image/', views.upload_image, name='upload_image'),
    path('delete/<int:image_id>/', views.delete_image, name='delete_image'),
    path('edit/<int:image_id>/', views.edit_image, name='edit_image'),
    path('send-chat/', views.send_chat, name='send_chat'),
]