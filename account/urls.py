from django.urls import path
from django.contrib.auth.views import LoginView, LogoutView
from .views import signup_view  # Assuming you already created this

urlpatterns = [
    path('signup/', signup_view, name='signup'),
    path('login/', LoginView.as_view(template_name='account/login.html'), name='login'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
]
