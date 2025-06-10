import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests

# Fungsi untuk mengambil animasi Lottie dari URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Judul
st.set_page_config(page_title="Prediksi Kepribadian", page_icon="🧠", layout="centered")
st.title("🧠 Aplikasi Prediksi Kepribadian Manusia")
st.markdown("Selamat datang! Aplikasi ini memprediksi apakah kamu **Introvert** atau **Extrovert** berdasarkan kebiasaanmu.")

# Tambahkan animasi di bagian atas
intro_animation = load_lottie_url("https://lottie.host/76aa15bb-c8f2-4b7b-ae64-113cb15a3f3a/BTVPFk1O5A.json")
st_lottie(intro_animation, height=250)

st.markdown("---")

# Sidebar untuk input
st.sidebar.header("Masukkan Karakteristik Anda")
time_alone = st.sidebar.slider("Waktu yang dihabiskan sendirian (0-15)", 0, 15, 5)
stage_fear = st.sidebar.slider("Tingkat ketakutan berbicara di depan umum (0-15)", 0, 15, 5)
social_event = st.sidebar.slider("Frekuensi menghadiri acara sosial (0-15)", 0, 15, 5)
going_out = st.sidebar.slider("Frekuensi keluar rumah (0-15)", 0, 15, 5)
drained = st.sidebar.slider("Merasa lelah setelah bersosialisasi (0-15)", 0, 15, 5)
friends_circle = st.sidebar.slider("Ukuran lingkaran pertemanan (0-15)", 0, 15, 5)
post_frequency = st.sidebar.slider("Frekuensi memposting di media sosial (0-15)", 0, 15, 5)

# Button untuk prediksi
if st.sidebar.button("Prediksi"):
    input_data = pd.DataFrame({
        'Time_spent_Alone': [time_alone],
        'Stage_fear': [stage_fear],
        'Social_event_attendance': [social_event],
        'Going_outside': [going_out],
        'Drained_after_socializing': [drained],
        'Friends_circle_size': [friends_circle],
        'Post_frequency': [post_frequency]
    })

    prediction = model.predict(input_data)
    predicted_label = "Introvert" if prediction[0] == 0 else "Extrovert"

    st.markdown("---")
    st.subheader("🔍 Hasil Prediksi Anda:")

    if predicted_label == "Introvert":
        st_lottie(load_lottie_url("https://lottie.host/cf3dbf1b-bdc5-49e4-b845-5a6cfce169b1/fJNkivX7uI.json"), height=300)
        st.success("🌙 Kamu cenderung **Introvert**")
        st.markdown("> _Jalani hidup Anda seperti yang Anda inginkan, bukan seperti cara masyarakat memberi tahu Anda._")
    else:
        st_lottie(load_lottie_url("https://lottie.host/3dfae621-53c2-4e4d-bb2d-f89889e54e5f/1ALPM3Ct29.json"), height=300)
        st.success("🌞 Kamu cenderung **Extrovert**")
        st.markdown("> _Terkadang menjadi ekstrovert memanglah sangat menguntungkan._")

    st.markdown("---")
    st.info("Catatan: Prediksi ini berdasarkan data dan model Machine Learning. Gunakan sebagai referensi, bukan sebagai penilaian mutlak.")
