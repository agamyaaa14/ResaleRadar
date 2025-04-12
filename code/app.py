import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# Load model and dataset
model = CatBoostRegressor()
model.load_model("code/catboost_final_model.cbm")
df = pd.read_excel("dataset/cleaned_car_data.xlsx")

# Extract brand and model from 'name_model'
df['brand'] = df['name_model'].str.split().str[1]
df['model'] = df['name_model'].str.extract(rf'{df["brand"].iloc[0]}\s+(.*?)\s')

# Page settings
st.set_page_config(page_title="Resale Radar", layout="centered")
st.markdown("<h1 style='color:orange;'>🚗 Resale Radar</h1>", unsafe_allow_html=True)
st.write("### Predict your car’s resale value accurately and instantly!")

# Sidebar: mode switch
mode = st.sidebar.radio("Choose Input Mode", ["🔰 I don't know specs", "⚙️ I know specs"])

# Brand and Model Selection
brand = st.selectbox("Choose Car Brand", sorted(df['brand'].dropna().unique()))
filtered_models = df[df['brand'] == brand]['name_model'].dropna().unique()
model_selected = st.selectbox("Choose Model", filtered_models)

# Location
locations = df['car_location'].dropna().unique()
location = st.selectbox("Car Registration Location", sorted(locations))

# Transmission
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])

# Logic for average specs (used when user doesn’t know full specs)
model_data = df[df['name_model'] == model_selected]
model_mean = model_data.mean(numeric_only=True).to_dict()

# Mode 1: No detailed specs
if mode == "🔰 I don't know specs":
    st.markdown("##### We'll use intelligent defaults for your car’s specifications.")
    car_age = st.slider("Car Age (Years)", 0, 20, int(model_mean.get("car_age", 5)))
    km_driven = st.slider("KM Driven", 0, 300000, int(model_mean.get("km_driven", 50000)), step=5000)
    no_of_owner = st.selectbox("Number of Owners", sorted(df['no_of_owner'].dropna().unique()))

# Mode 2: Detailed input
else:
    car_age = st.number_input("Car Age (Years)", min_value=0, max_value=30, value=int(model_mean.get("car_age", 5)))
    km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=int(model_mean.get("km_driven", 50000)), step=5000)
    no_of_owner = st.selectbox("Number of Owners", sorted(df['no_of_owner'].dropna().unique()))
    fuel_tank_capacity = st.number_input("Fuel Tank Capacity (L)", value=round(model_mean.get("fuel_tank_capacity", 40.0), 1))
    displacement = st.number_input("Engine Displacement (cc)", value=int(model_mean.get("displacement", 1200)))
    mileage = st.number_input("Mileage (kmpl)", value=round(model_mean.get("mileage", 18.0), 1))
    bootspace = st.number_input("Bootspace (L)", value=int(model_mean.get("bootspace", 350)))
    seating_capacity = st.selectbox("Seating Capacity", [4, 5, 6, 7], index=1)
    avg_tyre_life = st.slider("Tyre Life Remaining (%)", 0, 100, int(model_mean.get("avg_tyre_life%", 80)))

# Common to both modes
price_segment = 'Mid-Range' if 'Mid' in model_selected or model_mean.get("price_segment") == "Mid-Range" else 'Budget'
fuel_tank_capacity = fuel_tank_capacity if mode == "⚙️ I know specs" else model_mean.get("fuel_tank_capacity", 40.0)
displacement = displacement if mode == "⚙️ I know specs" else model_mean.get("displacement", 1200)
mileage = mileage if mode == "⚙️ I know specs" else model_mean.get("mileage", 18.0)
bootspace = bootspace if mode == "⚙️ I know specs" else model_mean.get("bootspace", 350)
seating_capacity = seating_capacity if mode == "⚙️ I know specs" else 5
avg_tyre_life = avg_tyre_life if mode == "⚙️ I know specs" else model_mean.get("avg_tyre_life%", 80)

# Derived features
bootspace_per_seat = bootspace / seating_capacity
mileage_per_cc = mileage / displacement

# Final input for prediction
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

# Predict
log_price = model.predict(input_data)
predicted_price = np.expm1(log_price)[0]

# Output
st.success(f"### 💰 Estimated Resale Price: ₹{predicted_price:,.0f}")

# Feature Importance Plot
if st.checkbox("📊 Show Feature Importance"):
    importances = model.get_feature_importance(prettified=True)
    fig, ax = plt.subplots()
    importances.sort_values(by='Importances', ascending=True).plot.barh(x='Feature Id', y='Importances', ax=ax, color='orange', legend=False)
    ax.set_title("🔍 Feature Importance")
    st.pyplot(fig)