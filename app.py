# app.py
import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import pickle
import json
import requests
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Prediksi Kepribadian", layout="centered")

# Fungsi load animasi dari URL

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animasi dari LottieFiles
intro_animation = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json")

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Header aplikasi
st.title("Prediksi Tipe Kepribadian Anda")

# Tampilkan animasi
st_lottie(intro_animation, height=250)

st.markdown("Silakan sesuaikan preferensi Anda di bawah ini:")

# Sidebar input
Time_spent_Alone = st.slider("Waktu sendirian (0-15)", 0, 15, 5)
Stage_fear = st.slider("Ketakutan tampil di depan umum (0-15)", 0, 15, 5)
Social_event_attendance = st.slider("Kehadiran di acara sosial (0-15)", 0, 15, 5)
Going_outside = st.slider("Frekuensi keluar rumah (0-15)", 0, 15, 5)
Drained_after_socializing = st.slider("Kelelahan setelah bersosialisasi (0-15)", 0, 15, 5)
Friends_circle_size = st.slider("Ukuran lingkaran pertemanan (0-15)", 0, 15, 5)
Post_frequency = st.slider("Frekuensi posting di media sosial (0-15)", 0, 15, 5)

# Prediksi
if st.button("Prediksi Kepribadian"):
    input_data = pd.DataFrame({
        "Time_spent_Alone": [Time_spent_Alone],
        "Stage_fear": [Stage_fear],
        "Social_event_attendance": [Social_event_attendance],
        "Going_outside": [Going_outside],
        "Drained_after_socializing": [Drained_after_socializing],
        "Friends_circle_size": [Friends_circle_size],
        "Post_frequency": [Post_frequency]
    })

    result = model.predict(input_data)[0]
    if result == 0:
        st.success("Hasil Prediksi: **Introvert**")
        st.markdown("> _\"Jalani hidup Anda seperti yang Anda inginkan, bukan seperti cara masyarakat memberi tahu Anda.\"_")
        st.image("https://cdn-icons-png.flaticon.com/512/1864/1864514.png", width=120)
    else:
        st.success("Hasil Prediksi: **Extrovert**")
        st.markdown("> _\"Terkadang menjadi ekstrovert memanglah sangat menguntungkan.\"_")
        st.image("https://cdn-icons-png.flaticon.com/512/1864/1864517.png", width=120)

st.markdown("\n---\n")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit dan Machine Learning")
