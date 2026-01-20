%%writefile streamlit_app.py
import streamlit as st
import pickle
import numpy as np

# Load model dan scaler
with open("model_air.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler_air.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Water Quality Predictor", page_icon="💧")
st.title("🌊 Prediksi Kelayakan Air Minum")
st.write("Aplikasi ini memprediksi apakah air layak minum berdasarkan parameter kimia.")

# Form Input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        ph = st.number_input("pH", value=7.0)
        hardness = st.number_input("Hardness", value=200.0)
        solids = st.number_input("Solids", value=20000.0)
        chloramines = st.number_input("Chloramines", value=7.0)
    with col2:
        sulfate = st.number_input("Sulfate", value=300.0)
        conductivity = st.number_input("Conductivity", value=400.0)
        organic_carbon = st.number_input("Organic Carbon", value=15.0)
        trihalomethanes = st.number_input("Trihalomethanes", value=60.0)
    
    turbidity = st.number_input("Turbidity", value=4.0)
    submit = st.form_submit_button("Prediksi Sekarang")

if submit:
    # Urutan fitur harus sama dengan saat training
    features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Scaling & Prediksi
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    if prediction[0] == 1:
        st.success("✅ HASIL: Air ini LAYAK untuk diminum.")
    else:
        st.error("❌ HASIL: Air ini TIDAK LAYAK untuk diminum.")