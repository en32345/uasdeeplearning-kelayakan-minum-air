import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Kelayakan Air", layout="wide")

st.title("💧 Aplikasi Prediksi Kelayakan Minum Air")
st.write("Aplikasi ini memprediksi apakah air aman untuk dikonsumsi berdasarkan parameter kualitas air.")

# --- LOAD & PREPROCESS DATA ---
@st.cache_resource
def train_model():
    # Membaca dataset
    df = pd.read_csv('water_potability.csv')
    
    # Menangani missing values dengan mengisi nilai rata-rata (mean)
    df['ph'] = df['ph'].fillna(df['ph'].mean())
    df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean())
    
    # Memisahkan fitur dan target
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Split data (untuk mengecek akurasi, meskipun di aplikasi kita pakai seluruh data untuk train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan training model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# Load model dan akurasi
model, accuracy = train_model()

# --- INPUT FORM ---
st.subheader("Masukkan Parameter Kualitas Air")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        ph = st.number_input("pH (0-14)", 0.0, 14.0, 7.0)
        hardness = st.number_input("Hardness (mg/L)", value=200.0)
        solids = st.number_input("Solids (ppm)", value=20000.0)
        chloramines = st.number_input("Chloramines (ppm)", value=7.0)
    with col2:
        sulfate = st.number_input("Sulfate (mg/L)", value=300.0)
        conductivity = st.number_input("Conductivity (μS/cm)", value=400.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", value=15.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=60.0)

    turbidity = st.number_input("Turbidity (NTU)", value=4.0)

# --- PREDIKSI ---
st.markdown("---")
if st.button("Cek Kelayakan Air", type="primary"):
    # Menyiapkan data untuk prediksi
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("✅ **Hasil: Layak Dimum (Potable)**")
        st.write(f"Keyakinan Model: {probability[0][1]*100:.2f}%")
    else:
        st.error("❌ **Hasil: Tidak Layak Minum (Not Potable)**")
        st.write(f"Keyakinan Model: {probability[0][0]*100:.2f}%")

# --- INFORMASI MODEL ---
with st.expander("Informasi Model"):
    st.write(f"Model yang digunakan: **Random Forest Classifier**")
    st.write(f"Akurasi Pengujian Model: **{accuracy*100:.2f}%**")
    st.write("Catatan: Data yang kosong pada dataset asli telah diisi dengan nilai rata-rata (mean imputing).")