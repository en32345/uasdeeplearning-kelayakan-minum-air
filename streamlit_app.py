import streamlit as st
import pickle
import numpy as np

# 1. Load Model dan Scaler
# Pastikan nama file ini sama dengan yang kamu upload ke GitHub
with open("model_air.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler_air.pkl", "rb") as f:
    scaler = pickle.load(f)

# 2. Judul Web
st.title("🌊 Prediksi Kelayakan Air Minum")
st.write("Aplikasi ini menggunakan Machine Learning untuk memprediksi kualitas air.")

# 3. Form Input
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        ph = st.number_input("pH", value=7.0)
        hardness = st.number_input("Hardness", value=150.0)
        solids = st.number_input("Solids", value=20000.0)
        chloramines = st.number_input("Chloramines", value=7.0)
    with col2:
        sulfate = st.number_input("Sulfate", value=300.0)
        conductivity = st.number_input("Conductivity", value=400.0)
        organic_carbon = st.number_input("Organic Carbon", value=15.0)
        trihalomethanes = st.number_input("Trihalomethanes", value=60.0)
    
    turbidity = st.number_input("Turbidity", value=4.0)
    submitted = st.form_submit_button("Prediksi Sekarang")

# 4. Logika Prediksi
if submitted:
    # Urutan data sesuai training
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Scaling
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.success("✅ HASIL: Air ini LAYAK untuk dikonsumsi (Potable)")
    else:
        st.error("❌ HASIL: Air ini TIDAK LAYAK untuk dikonsumsi (Not Potable)")