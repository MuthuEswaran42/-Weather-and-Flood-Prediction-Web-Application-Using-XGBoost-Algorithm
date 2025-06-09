from django.shortcuts import render, redirect
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from datetime import datetime, timedelta
from .models import ImageModel
from .forms import ImageUploadForm
from xgboost import XGBRegressor, XGBClassifier
from django.http import JsonResponse
import pytz
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect, get_object_or_404



# Weather API Setup
API_KEY = 'dc671ce4bced13b6a9eae6f7a12b0521'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Function to fetch current weather
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed'],
        'clouds': data['clouds']['all'],
        'visibility': data['visibility'],
    }

# Function to read historical weather data
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.drop_duplicates()
    return df

# Prepare data for Rain prediction
def prepare_data(data):
    le = LabelEncoder()
    data['windGustDir'] = le.fit_transform(data['windGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    x = data[['MinTemp', 'MaxTemp', 'windGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return x, y, le

# Train Rain prediction model
def train_rain_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"Mean Squared Error for Rain Model: {mse:.4f}")
    print(f"Accuracy of Rain Model: {acc*100:.2f}%")  # Accuracy in %

    return model

# Prepare data for regression models
def prepare_regression_data(data, feature):
    x, y = [], []
    for i in range(len(data) - 1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    return x, y

# Train generic regression model
def train_regression_model(x, y):
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(x, y)
    return model

# Predict future values
def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(int(next_value[0]))
    return predictions[1:]

# Read and prepare historical rainfall data
def read_rainfall_data(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df = df.rename(columns={'Rainfall': 'Rainfall_mm'})
    return df

# Prepare rainfall data for specific city
def prepare_city_data(df, city):
    city_df = df[df['city'].str.lower() == city.lower()]
    city_df = city_df.sort_values('Date')
    X = city_df[['month', 'year']]
    y = city_df['Rainfall_mm']
    last_date = city_df['Date'].max()
    return X, y, last_date

# Train rainfall prediction model
def train_rainfall_model(X, y):
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

# Predict rainfall for future months
def predict_rainfall(model, last_date, months=5):
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months + 1)]
    future_data = pd.DataFrame({
        'month': [d.month for d in future_dates],
        'year': [d.year for d in future_dates]
    })
    predictions = model.predict(future_data)
    return list(zip(future_dates, predictions))

# Function to assess flood risk based on rainfall
def flood_risk_assessment(rainfalls):
    risk = []
    for rain in rainfalls:
        if rain > 500:
            risk.append("High")
        elif rain > 300:
            risk.append("Moderate")
        else:
            risk.append("Low")
    return risk

# Main weather view function
def home_views(request):
    city = 'chennai'
    if request.method == 'POST':
        city = request.POST.get('city', 'chennai')  # Default to London if no city is entered
    
    # Get current weather data
    current_weather = get_current_weather(city)
    
    # Rain prediction part
    historical_data = read_historical_data('weather.csv')
    x, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(x, y)

    # Compass direction calculation
    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75), ("N", 348.75, 360)
    ]
    compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
    compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

    # Prepare data for current prediction
    current_data = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'windGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather['Wind_Gust_Speed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temp'],
    }
    current_df = pd.DataFrame([current_data])
    rain_prediction = rain_model.predict(current_df)[0]

    # Temperature and Humidity prediction
    x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
    temp_model = train_regression_model(x_temp, y_temp)
    hum_model = train_regression_model(x_hum, y_hum)

    future_temp = predict_future(temp_model, current_weather['temp_min'])
    future_humidity = predict_future(hum_model, current_weather['humidity'])

    # Rainfall prediction part
    rainfall_data = read_rainfall_data('final_one_day_per_month_rainfall_1.csv')
    city_data = prepare_city_data(rainfall_data, city)
    rainfall_model = train_rainfall_model(*city_data[:2])

    # Predict next 6 months of rainfall
    predicted_rainfall = predict_rainfall(rainfall_model, city_data[2])
    rainfalls = [int(prediction[1]) for prediction in predicted_rainfall]
    flood_risk = flood_risk_assessment(rainfalls)


    timezone = pytz.timezone('Asia/Kolkata')
    now = datetime.now(timezone)
    next_hour = now + timedelta(days=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(now + timedelta(days=i)).strftime("%d %b") for i in range(1, 6)]
    future_months = [(now + timedelta(days=31 * i)).strftime("%B") for i in range(1, 6)]
        


    time1, time2, time3, time4, time5 = future_times
    temp1, temp2, temp3, temp4, temp5 = future_temp
    hum1, hum2, hum3, hum4, hum5 = future_humidity
    rainfall1, rainfall2, rainfall3, rainfall4, rainfall5 = rainfalls
    mon1, mon2, mon3, mon4, mon5 = future_months
    flood1,flood2,flood3,flood4,flood5 = flood_risk 
        

    context = {
            'location': city,
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'Humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],
            'time': datetime.now(),
            'date': datetime.now().strftime("%d %m %y"),
            'wind': current_weather['Wind_Gust_Speed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['visibility'],
            'time1': time1, 'time2': time2, 'time3': time3, 'time4': time4, 'time5': time5,
            'mon1': mon1, 'mon2': mon2, 'mon3': mon3, 'mon4': mon4, 'mon5': mon5,
            'temp1': f"{round(temp1, 1)}", 'temp2': f"{round(temp2, 1)}", 'temp3': f"{round(temp3, 1)}",
            'temp4': f"{round(temp4, 1)}", 'temp5': f"{round(temp5, 1)}",
            'hum1': f"{round(hum1, 1)}", 'hum2': f"{round(hum2, 1)}", 'hum3': f"{round(hum3, 1)}",
            'hum4': f"{round(hum4, 1)}", 'hum5': f"{round(hum5, 1)}",
            'rainfall1': f"{round(rainfall1, 1)}", 'rainfall2': f"{round(rainfall2, 1)}", 'rainfall3': f"{round(rainfall3, 1)}",
            'rainfall4': f"{round(rainfall4, 1)}", 'rainfall5': f"{round(rainfall5, 1)}",
            'hum1': f"{round(hum1, 1)}", 'hum2': f"{round(hum2, 1)}", 'hum3': f"{round(hum3, 1)}",
            'hum4': f"{round(hum4, 1)}", 'hum5': f"{round(hum5, 1)}",
            #'flood1': f"{round(flood1, 1)}", 'flood2': f"{round(flood2, 1)}", 'flood3': f"{round(flood3, 1)}",
            #'flood4': f"{round(flood4, 1)}", 'flood5': f"{round(flood5, 1)}",
            'flood1': flood1, 'flood2': flood2, 'flood3': flood3, 'flood4': flood4, 'flood5': flood5,


            
        }

    return render(request, 'WF/home.html', context)

    return render(request, 'WF/home.html')

def about(request):
    images = ImageModel.objects.all()

    # Handle form submission here directly
    if request.method == 'POST':
        if request.user.is_authenticated:
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                image = form.save(commit=False)
                image.user = request.user
                image.save()
                return redirect('about')
        else:
            return redirect('login')

    else:
        form = ImageUploadForm()

    return render(request, 'WF/about.html', {'images': images, 'form': form})

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded file
        uploaded_image = request.FILES['image']
        link_url = request.POST.get('linkUrl', '')
        desc = request.POST.get('desc', '')

        # Save the image to the filesystem
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        file_url = fs.url(filename)

        # Save image metadata to the database
        new_image = ImageModel(image=file_url, link=link_url, description=desc, user=request.user)
        new_image.save()

        # Return a JSON response
        return JsonResponse({'message': 'Image uploaded successfully!'})

    return JsonResponse({'message': 'Failed to upload image.'}, status=400)

def delete_image(request, image_id):
    image = get_object_or_404(ImageModel, id=image_id)
    if request.user == image.user or request.user.is_superuser:
        image.delete()
        return redirect('about')  # or wherever your image list is shown
    else:
        return HttpResponseForbidden("You are not allowed to delete this image.")

def edit_image(request, image_id):
    image = get_object_or_404(ImageModel, id=image_id)
    if request.user != image.user and not request.user.is_superuser:
        return HttpResponseForbidden("You are not allowed to edit this image.")

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES, instance=image)
        if form.is_valid():
            form.save()
            return redirect('about')
    else:
        form = ImageUploadForm(instance=image)
    
    return render(request, 'WF/edit.html', {'form': form, 'image': image})
def send_chat(request):
    if request.method == "POST":
        data = json.loads(request.body)
        name = data.get('name')
        issue = data.get('issue')
        phone = data.get('phone')
        address = data.get('address')

        # Here, send SMS to admin using SMS API (like Twilio, Fast2SMS, etc.)
        print(name, issue, phone, address)  # or save to DB

        return JsonResponse({'status': 'success'})