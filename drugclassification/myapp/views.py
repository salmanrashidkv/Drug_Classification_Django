# myapp/views.py
import os
from django.shortcuts import render
from django.http import HttpResponse
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

def homepage(request):
    return render(request, "index.html")

def predict(request):
    if request.method == 'POST':
        try:
            # reading the inputs given by the user
            gender = request.POST['gender']
            age = int(request.POST['age'])
            bp = request.POST['bp']
            cholesterol = request.POST['cholesterol']
            na_to_k = int(request.POST['na_to_k'])

            # Encoding 'gender' using LabelEncoder
            label_encoder = LabelEncoder()
            gender_encoded = label_encoder.fit_transform([gender])[0]

            input_data = np.array([[gender_encoded, age, bp, cholesterol, na_to_k]])

            # Load the scaler using joblib
            scaler = joblib.load('myapp/final_Scaler.pkl')

            # Load the model using joblib
            model = joblib.load('myapp/final_Model.pkl')

            # Transform input data using the loaded scaler
            scaler_data = scaler.transform(input_data)

            # Make predictions using the loaded model
            prediction = model.predict(scaler_data)

            # Showing the prediction results in a UI
            return render(request, 'results.html', {'prediction': prediction[0]})
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return HttpResponse('Something went wrong')
    else:
        return render(request, 'index.html')
