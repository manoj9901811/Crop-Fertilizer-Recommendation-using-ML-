from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.http import HttpResponseRedirect
from django.contrib import messages
from .models import Products,Crop

# Create your views here.

def index(request):
    return render(request,"index.html")

def contact(request):
    return render(request,"contact.html")

def about(request):
    return render(request,"about.html")

def service(request):
    return render(request,"service.html")


def register(request):
    if request.method=="POST":
        fn=request.POST['fname']
        ln=request.POST['lname']
        un=request.POST['uname']
        em=request.POST['email']
        ps=request.POST['psw']
        ps1=request.POST['psw1']
        if ps==ps1:
            if User.objects.filter(username=un).exists():
                messages.info(request,"Username Exists")
                return render(request,"register.html")
            elif User.objects.filter(email=em).exists():
                messages.info(request,"Email Exists")
                return render(request,"register.html")
            else:

                #Store value in database
                #Create object for the table name
                user=User.objects.create_user(first_name=fn,
        last_name=ln,username=un,email=em,password=ps1)
                user.save()
                return HttpResponseRedirect('login')
        else:
            messages.info(request,"Password Not Matching")
            return render(request,"register.html")
    else:
        return render(request,"register.html")
    return render(request,"register.html")


def login(request):
    if request.method=="POST":
        un=request.POST['uname']
        ps=request.POST['psw']
        user=auth.authenticate(username=un,password=ps)
        if user is not None:
            auth.login(request,user)
            return HttpResponseRedirect('/farmer/index')
        else:
            messages.info(request,"Invalid Credentials")
            return render(request,"login.html")

    return render(request,"login.html")


def logout(request):
    auth.logout(request)
    return HttpResponseRedirect('/farmer/index')


def product(request):
    p=Products.objects.all()
    return render(request,"product.html",{"p":p})

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from django.shortcuts import render
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from .models import Crop

def crop(request):
    if request.method == "POST":
        # Getting Input Data
        nitro = int(request.POST['Nitrogen'])
        phos = int(request.POST['Phosphorous'])
        potas = int(request.POST['Potassium'])
        temp = float(request.POST['Temperature'])
        humid = float(request.POST['Humidity'])
        ph = float(request.POST['Ph'])
        rain = float(request.POST['Rainfall'])

        # Load Dataset
        df = pd.read_csv('static/Crop_recommendation.csv')

        # Data Preprocessing
        df.dropna(inplace=True)
        X = df.drop("label", axis=1)
        y = df["label"]

        # Train Model
        log = LogisticRegression()
        log.fit(X, y)

        # Make Prediction
        crop_data = np.array([[nitro, phos, potas, temp, humid, ph, rain]], dtype=object)
        pred_crop = log.predict(crop_data)[0]  # Get the first predicted value

        # Save Prediction in Database
        crop = Crop.objects.create(
            Nitrogen=nitro, Phosphorous=phos, Potassium=potas,
            Temperature=temp, Rainfall=rain, PH=ph,
            Humidity=humid, Recommend_Crop=pred_crop
        )
        crop.save()

        # Generate Data Visualizations
        plots = generate_visualizations(df)

        return render(request, "crop.html",{ "prediction": pred_crop,
            "plots": plots})

    return render(request, "crop.html")

def generate_visualizations(df):
    """ Generate and save plots as Base64 images to display in the Django template """

    plots = {}

    # Plot 1: Histogram of Nitrogen, Phosphorous, and Potassium
    plt.figure(figsize=(8, 4))
    sns.histplot(df['N'], bins=30, kde=True, color='blue', label="Nitrogen")
    sns.histplot(df['P'], bins=30, kde=True, color='red', label="Phosphorous")
    sns.histplot(df['K'], bins=30, kde=True, color='green', label="Potassium")
    plt.legend()
    plots['nutrients_histogram'] = save_plot_as_base64()

    # Plot 2: Temperature vs. Humidity Scatter Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df["temperature"], y=df["humidity"], hue=df["label"], palette="viridis", alpha=0.6)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.title("Temperature vs. Humidity")
    plots['temp_humidity_scatter'] = save_plot_as_base64()

    # Plot 3: Crop Distribution Count
    plt.figure(figsize=(8, 8))
    crop_counts = df["label"].value_counts()
    plt.pie(crop_counts, labels=crop_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
    plt.title("Crop Distribution")
    plots['crop_piechart'] = save_plot_as_base64()

    return plots

def save_plot_as_base64():
    """ Save the current Matplotlib figure as a Base64 string """
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{image_base64}"

def predict(request):
    return render(request,"predict.html")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from django.shortcuts import render
from sklearn.linear_model import LogisticRegression
from Farmer.models import Fertilizer  # Ensure this model exists

def fertilizer(request):
    prediction = None
    error_message = None
    plots = {}  # ✅ Initialize plots to avoid UnboundLocalError

    if request.method == "POST":
        try:
            # Get input data
            temp = float(request.POST['Temperature'])
            humid = float(request.POST['Humidity'])
            moist = float(request.POST['Moisture'])
            soil_type = request.POST['Soil_Type']
            crop_type = request.POST['Crop_Type']
            nitro = int(request.POST['Nitrogen'])
            phos = int(request.POST['Phosphorous'])
            potas = int(request.POST['Potassium'])

            # Load dataset
            df = pd.read_csv('static/Fertilizer.csv')

            # Ensure dataset is not empty
            if df.empty:
                raise ValueError("Dataset is empty! Please check the CSV file.")

            # Ensure categorical columns are string
            df["Soil Type"] = df["Soil Type"].astype(str)
            df["Crop Type"] = df["Crop Type"].astype(str)

            # Unique values in dataset
            soil_types = df["Soil Type"].unique()
            crop_types = df["Crop Type"].unique()

            # Ensure user input matches available values
            if soil_type not in soil_types:
                raise ValueError(f"Invalid Soil Type: {soil_type}. Choose from {list(soil_types)}")
            if crop_type not in crop_types:
                raise ValueError(f"Invalid Crop Type: {crop_type}. Choose from {list(crop_types)}")

            # Convert categorical values to numerical codes
            soil_mapping = {val: idx for idx, val in enumerate(soil_types)}
            crop_mapping = {val: idx for idx, val in enumerate(crop_types)}

            df["Soil Type"] = df["Soil Type"].map(soil_mapping)
            df["Crop Type"] = df["Crop Type"].map(crop_mapping)

            # Get numerical codes for input values
            soil_code = soil_mapping[soil_type]
            crop_code = crop_mapping[crop_type]

            # Prepare training data
            X = df.drop("Fertilizer Name", axis=1)
            y = df["Fertilizer Name"]

            # Train Model
            model = LogisticRegression()
            model.fit(X, y)

            # Make Prediction
            fert_data = np.array([[temp, humid, moist, soil_code, crop_code, nitro, phos, potas]], dtype=object)
            prediction = model.predict(fert_data)[0]

            # Save prediction to database
            fert = Fertilizer.objects.create(
                Temperature=temp, Humidity=humid, Moisture=moist,
                Soil_Type=soil_type, Crop_Type=crop_type,
                Nitrogen=nitro, Phosphorous=phos, Potassium=potas,
                Recommended_Fertilizer=prediction
            )
            fert.save()

            # ✅ Generate Plots
            print("Generating plots...")  # Debugging line
            plots = generate_fertilizer_visualizations(df)
            print("Plots generated:", plots)  # Debugging line

        except Exception as e:
            error_message = str(e)
            print("Error:", error_message)  # Debugging line

    return render(request, "fertilizer.html", {
        "prediction": prediction,
        "error_message": error_message,
        "plots": plots  # ✅ Always defined
    })


def generate_fertilizer_visualizations(df):
    """Generate and save plots as Base64 images to display in the Django template."""
    plots = {}

    try:
        print("🔹 Generating Nutrient Histogram...")
        plt.figure(figsize=(8, 4))
        sns.histplot(df['Nitrogen'], bins=30, kde=True, color='blue', label="Nitrogen")
        sns.histplot(df['Phosphorous'], bins=30, kde=True, color='red', label="Phosphorous")
        sns.histplot(df['Potassium'], bins=30, kde=True, color='green', label="Potassium")
        plt.legend()
        plt.title("Nutrient Distribution")
        plots['nutrient_histogram'] = save_plot_as_base64()
        print("✅ Nutrient Histogram Generated!")

        print("🔹 Generating Moisture vs Humidity Scatter Plot...")
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df["Moisture"], y=df["Humidity"], hue=df["Fertilizer Name"], palette="viridis", alpha=0.6)
        plt.xlabel("Moisture")
        plt.ylabel("Humidity")
        plt.title("Moisture vs Humidity (Fertilizer-wise)")
        plots['moisture_humidity_scatter'] = save_plot_as_base64()
        print("✅ Moisture vs Humidity Scatter Plot Generated!")

        print("🔹 Generating Fertilizer Usage Pie Chart...")
        plt.figure(figsize=(8, 8))
        fert_counts = df["Fertilizer Name"].value_counts()
        plt.pie(fert_counts, labels=fert_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
        plt.title("Fertilizer Usage Distribution")
        plots['fertilizer_piechart'] = save_plot_as_base64()
        print("✅ Fertilizer Usage Pie Chart Generated!")

    except Exception as e:
        print("🚨 Error in generate_fertilizer_visualizations:", str(e))

    return plots

import io
import base64
import matplotlib.pyplot as plt

def save_plot_as_base64():
    """Convert Matplotlib figure to Base64-encoded image."""
    try:
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close('all')  # ✅ Ensure figure is cleared after saving
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print("Error in save_plot_as_base64:", str(e))  # Debugging
        return None  # Return None if error occurs

