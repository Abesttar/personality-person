import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Judul aplikasi
st.title("Prediksi Kepribadian: Introvert atau Extrovert")

# Load model dan encoder
try:
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("❌ File model.pkl atau label_encoder.pkl tidak ditemukan. Pastikan file sudah diunggah.")
    st.stop()

# Fitur yang digunakan untuk prediksi
features = [
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]

# Sidebar untuk input pengguna
st.sidebar.header("Input Karakteristik Pribadi")
user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.slider(
        feature.replace("_", " "), min_value=0, max_value=20, value=5
    )

# Ubah input ke DataFrame
input_df = pd.DataFrame([user_input])

# Tampilkan input pengguna
st.subheader("Data Input")
st.write(input_df)

# Prediksi dengan model
prediction = model.predict(input_df)[0]
predicted_label = le.inverse_transform([prediction])[0]

# Tampilkan hasil prediksi
st.subheader("Hasil Prediksi")
if predicted_label == "Introvert":
    st.success("🧠 Hasil Prediksi: Kamu cenderung **Introvert**.")
else:
    st.success("🎉 Hasil Prediksi: Kamu cenderung **Extrovert**.")
