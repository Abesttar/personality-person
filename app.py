import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# Fungsi untuk memuat animasi dari URL
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Animasi Lottie
intro_animation = load_lottie_url("https://lottie.host/531f8358-95c0-48e9-8029-9385c3e2b82a/dwPDsRfCgG.json")
st_lottie(intro_animation, height=250)

st.title("Prediksi Kepribadian: Introvert atau Extrovert")

st.markdown("""
Gunakan slider di bawah ini untuk memasukkan kebiasaan dan preferensi Anda, lalu klik "Prediksi" untuk mengetahui kepribadian Anda berdasarkan model machine learning.
""")

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Sidebar input
st.sidebar.header("Input Fitur Pengguna")
time_alone = st.sidebar.slider("Waktu dihabiskan sendiri (0-15)", 0, 15, 5)
stage_fear = st.sidebar.slider("Tingkat ketakutan tampil di depan umum (0-15)", 0, 15, 5)
social_event = st.sidebar.slider("Kehadiran di acara sosial (0-15)", 0, 15, 5)
going_out = st.sidebar.slider("Frekuensi keluar rumah (0-15)", 0, 15, 5)
drain_social = st.sidebar.slider("Tingkat lelah setelah sosialisasi (0-15)", 0, 15, 5)
friends_size = st.sidebar.slider("Ukuran lingkaran pertemanan (0-15)", 0, 15, 5)
post_freq = st.sidebar.slider("Frekuensi memposting di media sosial (0-15)", 0, 15, 5)

# Dataframe input pengguna
user_input = pd.DataFrame({
    'Time_spent_Alone': [time_alone],
    'Stage_fear': [stage_fear],
    'Social_event_attendance': [social_event],
    'Going_outside': [going_out],
    'Drained_after_socializing': [drain_social],
    'Friends_circle_size': [friends_size],
    'Post_frequency': [post_freq]
})

# Tombol prediksi
if st.button("Prediksi"):
    result = model.predict(user_input)[0]

    if result == 0:
        st.subheader("Hasil Prediksi: Introvert 🧘")
        st.info("\"Jalani hidup Anda seperti yang Anda inginkan, bukan seperti cara masyarakat memberi tahu Anda.\"")
        st.image("https://cdn-icons-png.flaticon.com/512/3606/3606750.png", width=150)
    else:
        st.subheader("Hasil Prediksi: Extrovert 🗣️")
        st.success("\"Terkadang menjadi ekstrovert memanglah sangat menguntungkan.\"")
        st.image("https://cdn-icons-png.flaticon.com/512/3606/3606762.png", width=150)

    st.markdown("Terima kasih telah menggunakan aplikasi prediksi kepribadian ini! 🙌")
