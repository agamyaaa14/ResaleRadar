import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #0f1117;
        color: white;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label, .stTextInput label {
        color: white !important;
        font-weight: 600;
        font-size: 16px;
    }
    .stApp {
        background-color: #0f1117;
        color: white;
    }
    .css-1v0mbdj, .css-1avcm0n { 
        background-color: #0f1117 !important; 
        color: white !important; 
    }
    .stButton button {
        color: black !important;
        font-weight: bold;
        background-color: #fc5e0f;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    h1, h2, h3, h4, h5 {
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and dataset
model = CatBoostRegressor()
model.load_model("code/catboost_final_model.cbm")
df = pd.read_excel("dataset/cleaned_car_data.xlsx")

# Extract brand and model
df['brand'] = df['name_model'].str.split().str[1]
df['model'] = df['name_model'].str.extract(rf'{df["brand"].iloc[0]}\s+(.*?)\s')

# Logo/Slogan
st.image("images/rr1.png", use_container_width=True)

mode = st.sidebar.radio("Choose Input Mode", ["I don't know specs", "I know specs"])

# Brand and Model
st.markdown("##### Choose your car brand and model:")
brand = st.selectbox("Car Brand", sorted(df['brand'].dropna().unique()))
filtered_models = df[df['brand'] == brand]['name_model'].dropna().unique()
model_selected = st.selectbox("Car Model", filtered_models)

# Location (filtered)
st.markdown("##### Where is the car registered?")
locations = sorted(set(loc for loc in df['car_location'].dropna().unique() if len(loc) > 2))
location = st.selectbox("Registration Location", locations)

# Transmission
st.markdown("##### What is the transmission type?")
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])

# Default values based on selected model
model_data = df[df['name_model'] == model_selected]
model_mean = model_data.mean(numeric_only=True).to_dict()

# Mode 1: Intelligent Defaults
if mode == "I don't know specs":
    st.markdown("##### We'll use intelligent defaults for your car’s specifications.")
    car_age = st.slider("Car Age (Years)", 0, 20, value=0)
    km_driven = st.slider("KM Driven", 0, 300000, value=0, step=5000)
    no_of_owner = st.selectbox("Number of Owners", sorted(df['no_of_owner'].dropna().unique()))

# Mode 2: Manual Input
else:
    st.markdown("##### Enter detailed specifications:")
    car_age = st.number_input("Car Age (Years)", min_value=0, max_value=30, value=0)
    km_driven = st.slider("KM Driven", min_value=0, max_value=160000, value=48000, step=1000)
    no_of_owner = st.selectbox("Number of Owners", sorted(df['no_of_owner'].dropna().unique()))
    fuel_tank_capacity = st.slider("Fuel Tank Capacity (L)", min_value=20, max_value=85, value=44, step=1)
    displacement = st.slider("Engine Displacement (cc)", min_value=600, max_value=2800, value=1290, step=50)
    mileage = st.slider("Mileage (kmpl)", min_value=8.0, max_value=35.0, value=19.0, step=0.5)
    bootspace = st.slider("Bootspace (L)", min_value=100, max_value=800, value=346, step=10)
    seating_capacity = st.selectbox("Seating Capacity", [4, 5, 6, 7])
    avg_tyre_life = st.slider("Tyre Life Remaining (%)", 20, 100, value=65, step=5)

# Auto-fill if not manual mode
price_segment = 'Mid-Range' if 'Mid' in model_selected or model_mean.get("price_segment") == "Mid-Range" else 'Budget'
fuel_tank_capacity = fuel_tank_capacity if mode == "I know specs" else model_mean.get("fuel_tank_capacity", 40.0)
displacement = displacement if mode == "I know specs" else model_mean.get("displacement", 1200)
mileage = mileage if mode == "I know specs" else model_mean.get("mileage", 18.0)
bootspace = bootspace if mode == "I know specs" else model_mean.get("bootspace", 350)
seating_capacity = seating_capacity if mode == "I know specs" else 5
avg_tyre_life = avg_tyre_life if mode == "I know specs" else model_mean.get("avg_tyre_life%", 80)

# Derived features
bootspace_per_seat = bootspace / seating_capacity if seating_capacity else 0
mileage_per_cc = mileage / displacement if displacement else 0

# Button to predict
if st.button("Estimate Resale Price"):
    input_data = pd.DataFrame([{
        'brand': brand,
        'transmission': transmission,
        'car_location': location,
        'price_segment': price_segment,
        'km_driven': km_driven,
        'avg_tyre_life%': avg_tyre_life,
        'fuel_tank_capacity': fuel_tank_capacity,
        'displacement': displacement,
        'car_age': car_age,
        'no_of_owner': no_of_owner,
        'mileage_per_cc': mileage_per_cc,
        'bootspace_per_seat': bootspace_per_seat
    }])

    log_price = model.predict(input_data)
    predicted_price = np.expm1(log_price)[0]

    st.markdown(f"<h3 style='color:#fc5e0f;'>Estimated Resale Price: ₹{predicted_price:,.0f}</h3>", unsafe_allow_html=True)
    st.markdown("**Disclaimer:** This is a machine-generated estimate and may vary from actual market values.")
