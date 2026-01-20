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

# --- CUSTOM CSS UNTUK STYLE ---
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
    .stNumberInput {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_stdio=True)

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
except:
    st.error("Gagal memuat model. Pastikan file .pkl sudah diupload ke GitHub.")

# --- HEADER ---
st.image("https://img.freepik.com/free-vector/water-drop-with-medical-icons-vector_53876-175232.jpg", use_container_width=True)
st.title("🌊 Smart Water Analyzer")
st.subheader("Prediksi Kelayakan Air Minum dengan Machine Learning")
st.info("Masukkan parameter kimia air di bawah ini untuk mendapatkan analisis instan.")

# --- INPUT FORM ---
with st.container():
    st.write("### 🧪 Parameter Kimia Air")
    col1, col2 = st.columns(2)
    
    with col1:
        ph = st.number_input("pH (0-14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        hardness = st.number_input("Hardness (mg/L)", value=200.0)
        solids = st.number_input("Solids (ppm)", value=20000.0)
        chloramines = st.number_input("Chloramines (ppm)", value=7.0)
        sulfate = st.number_input("Sulfate (mg/L)", value=300.0)

    with col2:
        conductivity = st.number_input("Conductivity (μS/cm)", value=400.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", value=15.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=60.0)
        turbidity = st.number_input("Turbidity (NTU)", value=4.0)

# --- PROSES PREDIKSI ---
st.markdown("---")
if st.button("MULAI ANALISIS SISTEM"):
    with st.spinner('Sedang menghitung data...'):
        time.sleep(1.5) # Efek loading agar terlihat canggih
        
        # Susun Data
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
        
        # Scaling & Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # --- TAMPILAN HASIL ---
        if prediction == 1:
            st.balloons()
            st.success("### ✅ HASIL: AIR LAYAK MINUM")
            st.markdown("""
                **Rekomendasi:**
                - Parameter air berada dalam batas aman.
                - Aman untuk dikonsumsi harian.
            """)
        else:
            st.error("### ❌ HASIL: AIR TIDAK LAYAK")
            st.markdown("""
                **Peringatan:**
                - Terdeteksi parameter yang melebihi ambang batas kesehatan.
                - Perlu pengolahan lebih lanjut (filtrasi/perebusan).
            """)

# --- FOOTER ---
st.markdown("<br><hr><center>Dibuat untuk UAS Deep Learning © 2024</center>", unsafe_allow_html=True)