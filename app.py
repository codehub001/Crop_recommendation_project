# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect ,session,url_for,session
from markupsafe import Markup
import numpy as np
import pandas as pd


from disease import disease_dic
from fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from model import ResNet9


import requests
import config
import pickle
import io
# import torch
# from torchvision import transforms
from PIL import Image
# from utils.model import ResNet9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup
import base64
import requests
import sqlite3

from warnings import filterwarnings
filterwarnings('ignore')
# Load ML model
forest = pickle.load(open('models/yield_rf.pkl', 'rb'))  # yield
cp = pickle.load(open('models/forest.pkl', 'rb'))  # price
# ==============================================================================================
model = pickle.load(open('models/classifier.pkl','rb'))
ferti = pickle.load(open('models/xgb.pkl','rb'))


# Loading crop recommendation model


cr_xg = pickle.load(open('models/crop_xg.pkl', 'rb'))



disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    print('vgj,hDS|m n')
    print(response)

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

import requests

def get_weather(city):
    url = f"https://wttr.in/{city}?format=%C+|+%t+|+%h"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text.split(" | ")
        if len(data) == 3:
            condition, temperature, humidity = data
            return condition, temperature, humidity
    return None, None, None


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
app.secret_key ="1234567890"
# render home page

@app.route('/')
def index():
    return render_template('signup.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('signup.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            session['name'] = name
            return render_template('index.html')

    return render_template('signup.html')




@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('signup.html', msg='Successfully Registered')
    
    return render_template('signup.html')







@ app.route('/index.html')
def home():
    title = 'Crop harvest'
    return render_template('index.html', title=title)

# render crop recommendation form page

@ app.route('/stats', methods=['GET', 'POST'])
def stats():
    if request.method == 'POST':
        File = request.form['season']
        from check import check_stat
        highest_price_value,highest_price_day=check_stat(File)
        print(f"highest_price_day :\n\n{highest_price_day}\n\n")
        print(f"highest_price_value :\n\n{highest_price_value:.2f}\n\n")
        high_month=highest_price_day
        high_price=int(highest_price_value)
        session['priceday'] = high_month
        session['pricerate'] = high_price
        
        return render_template('price.html',high_price=high_price,high_month=high_month)

    return render_template('price.html')


import requests
from bs4 import BeautifulSoup








@ app.route('/weather', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':
        # try:
            
        city = request.form['city'].replace(" ", "+")  # Replace spaces with '+'
        
        condition, temperature, humidity = get_weather(city)


        if condition:
            print(f"City: {city.capitalize()}")
            print(f"Weather: {condition}")
            print(f"Temperature: {temperature}")
            print(f"Humidity: {humidity}")
            from datetime import datetime

            # Get the current time
            now = datetime.now()
            timee = now.strftime("%H:%M:%S")
            return render_template('weather.html', city=city.capitalize(), temp=temperature,hum=humidity, sky=condition,timee=timee)
        else:
            print("Invalid city name or data unavailable!")
            return render_template('weather.html',msg="Cant Fetch details for the Given input")

            
        # except:
        #     return render_template('weather.html', msg="Error while fetching")
    return render_template('weather.html')
    
    
@app.route('/disease_prediction', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        print(f"\n\n\n file name is  \n\n {file} \n\n\n")
        if not file:
            print("entered")
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
            

            prediction = Markup(str(disease_dic[prediction]))
            session['disease'] = prediction
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


@ app.route('/crop-recommend')
def crop_recommend():

    title = 'Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/yeild')
def yeild():
    title = 'crop yeild prediction'
    return render_template('crop_yeild.html', title=title )




@ app.route('/crop_predict', methods=['POST'])
def crop_predict():
    title = 'Crop Recommended'

    if request.method == 'POST':
        color = int(request.form.get('color'))
        Nitrogen = int(request.form.get('Nitrogen'))
        Phosphorus = int(request.form.get('Phosphorus'))
        Potassium = int(request.form.get('Potassium'))
        pH = float(request.form.get('pH'))
        Rainfall = float(request.form.get('Rainfall'))
        Temperature = float(request.form.get('Temperature'))

        # Prepare data
        data = np.array([[color, Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature]])
        print("Data for prediction:", data)

        
        my_prediction = cr_xg.predict(data)
        final_prediction = my_prediction[0]
        session['crop'] = final_prediction
        return render_template('crop-result.html', prediction=final_prediction, title=title)
        # else:
        #     return render_template('try_again.html', title=title)
# render fertilizer recommendation result page

@app.route('/fer_predict',methods=['POST'])
def fer_predict():
    color = request.form.get('color')
    Nitrogen = request.form.get('Nitrogen')
    Phosphorus = request.form.get('Phosphorus')
    Potassium = request.form.get('Potassium')
    pH = request.form.get('pH')
    Rainfall = request.form.get('Rainfall')
    Temperature = request.form.get('Temperature')
    Crop = request.form.get('crop')
    input = [float(color), float(Nitrogen), float(Phosphorus), float(Potassium), 
         float(pH), float(Rainfall), float(Temperature), float(Crop)]

# Make the input a 2D array (required by most models)
    input = np.array([input])

    # Correct way to call predict
    res = ferti.predict(input)[0]
    session['fer'] = res

    return render_template('fer_predict.html',res = res)
@ app.route('/yeild-predict', methods=['POST'])
def yeild_predict():
    title = 'yeild predicted'

    if request.method == 'POST':
        state = request.form['stt']
        district = request.form['city']
        year = 2024 #request.form['year']
        season = request.form['season']
        crop = request.form['crop']
        Temperature = request.form['Temperature']
        humidity = request.form['humidity']
        soilmoisture = request.form['soilmoisture']
        area = request.form['area']

        out_1 = forest.predict([[float(state),
                                 float(district),
                                 float(year),
                                 float(season),
                                 float(crop),
                                 float(Temperature),
                                 float(humidity),
                                 float(soilmoisture),
                                 float(area)]])
        if out_1[0] > 1000:
            ans=f" {out_1[0]:.2f} Kg"
        elif out_1[0] > 100:
            ans=f" {out_1[0]:.2f} Quintal"
        else:
            ans=f" {out_1[0]:.2f} Tons"

        session['yield'] = ans
        


        return render_template('yeild_prediction.html', prediction=ans, title=title)

    return render_template('try_again.html', title=title)


# render disease prediction result page

@ app.route('/price_predict', methods=['POST'])
def price_predict():
    title = 'price Suggestion'
    if request.method == 'POST':
        state = int(request.form['stt'])
        district = int(request.form['city'])
        year = int(request.form['year'])
        season = int(request.form['season'])
        crop = int(request.form['crop'])

        p_result = cp.predict([[float(state),
                                float(district),
                                float(year),
                                float(season),
                                float(crop)]])

        return render_template('price_prediction.html', title=title, p_result=p_result)
    return render_template('try_again.html',title=title)
                           

@app.route('/crop_price', methods=['GET', 'POST'])
def crop_price():
    # return "this is crop prediction page"
    title = 'crop price'
    return render_template('crop_price.html', title=title)

@app.route('/crop_fer', methods=['GET', 'POST'])
def crop_fer():
    title = 'crop Fertilizer'
    return render_template('fer.html', title=title) #,n=n,k=k,p=p,temp=temp,hum=hum,moi=moi)

@app.route('/prescription')
def prescription():
    return render_template('prescription.html')


@app.route('/logout')
def logout():
    return render_template('signup.html')
if __name__ == '__main__':
    app.run(debug=True)
