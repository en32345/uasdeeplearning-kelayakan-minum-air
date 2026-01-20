import streamlit as st
import pickle
import numpy as np
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Water Quality AI",
    page_icon="💧",
    layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True) # Sudah diperbaiki menjadi unsafe_allow_html

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    with open("model_air.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_air.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# --- TAMPILAN ATAS ---
st.title("🌊 Smart Water Analyzer")
st.info("Masukkan parameter kimia air untuk analisis kelayakan minum.")

# --- INPUT FORM ---
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        ph = st.number_input("pH", 0.0, 14.0, 7.0)
        hardness = st.number_input("Hardness", value=200.0)
        solids = st.number_input("Solids", value=20000.0)
        chloramines = st.number_input("Chloramines", value=7.0)
    with col2:
        sulfate = st.number_input("Sulfate", value=300.0)
        conductivity = st.number_input("Conductivity", value=400.0)
        organic_carbon = st.number_input("Organic Carbon", value=15.0)
        trihalomethanes = st.number_input("Trihalomethanes", value=60.0)

    turbidity = st.number_input("Turbidity", value=4.0)

# --- PREDIKSI ---
if st.button("MULAI ANALISIS"):
    with st.spinner('Menganalisis...'):
        time.sleep(1)
        # Urutan fitur sesuai training
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.balloons()
            st.success("### ✅ HASIL: AIR LAYAK MINUM")
        else:
            st.error("### ❌ HASIL: AIR TIDAK LAYAK")
