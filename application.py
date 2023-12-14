import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# import standard scaler and the machine llearning algorithm
Standardscaler=pickle.load(open('models/standardscaler.pkl','rb'))
model_prediction=pickle.load(open('models/modelforprediction.pkl','rb'))

# Make route for home page
'''
@app.route('/')
def index():
    return render_template ("index.html")

'''

@app.route('/',methods=['GET','POST'])
def Predict_model():
    if request.method=="POST":
        Pregnancies=float(request.form.get("Pregnancies"))
        Glucose=float(request.form.get("Glucose"))
        BloodPressure=float(request.form.get("BloodPressure"))
        SkinThickness=float(request.form.get("SkinThickness"))
        Insulin=float(request.form.get("Insulin"))
        BMI=float(request.form.get("BMI"))
        DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction"))
        Age=float(request.form.get('Age'))
        new_data_scaled = Standardscaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result=model_prediction.predict(new_data_scaled)
        return render_template('Result.html',result=result)

    else:
        return render_template("prediction_page.html")

'''
@app.route('/predictdata', methods=['GET', 'POST'])
def Predict_model():
    if request.method == "POST":
        Pregnancies = float(request.form.get("Pregnancies"))
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get('Age'))

        # Corrected line: pass numerical values, not feature names
        new_data_scaled = Standardscaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        result = model_prediction.predict(new_data_scaled)
        return render_template('prediction_page.html', result=result)

    else:
        return render_template("prediction_page.html")
'''

if __name__=="__main__":
    app.run(host="0.0.0.0")
