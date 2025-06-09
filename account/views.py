from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User
from django.shortcuts import redirect

# Login View
def login_view(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('home')  # Redirect to home after successful login
        else:
            # Handle failed login
            return render(request, 'WF/login.html', {'error': 'Invalid credentials'})
    
    return render(request, 'WF/login.html')

# Signup View
def signup_view(request):
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        
        if password1 == password2:
            user = User.objects.create_user(username=username, email=email, password=password1)
            login(request, user)  # Automatically log in the user after successful signup
            return redirect('home')
        else:
            # Handle mismatched passwords
            return render(request, 'account/signup.html', {'error': 'Passwords do not match'})
    
    return render(request, 'account/signup.html')
