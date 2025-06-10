import streamlit as st
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# === Fungsi untuk memuat animasi Lottie === #
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except:
        return None

# === Load model === #
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# === Judul & Animasi === #
st.set_page_config(page_title="Prediksi Kepribadian", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>🔮 Prediksi Kepribadian Kamu 🔍</h1>", unsafe_allow_html=True)
st.write("")

# === Tampilkan Lottie === #
with st.container():
    col1, col2 = st.columns([2, 3])
    with col1:
        lottie = load_lottie_url("https://lottie.host/1a81391d-671e-47ab-9023-c5e5e27e4691/XcVPuulNHE.json")
        if lottie:
            st_lottie(lottie, height=230)
    with col2:
        st.markdown("### 📌 Masukkan kebiasaan dan kepribadianmu:")
        st.write("Silakan isi beberapa pertanyaan berikut dan lihat apakah kamu termasuk **Introvert** atau **Ekstrovert**! 🚀")

# === Input User === #
st.markdown("---")
st.markdown("## 🧠 Jawab pertanyaan berikut:")

col1, col2 = st.columns(2)

with col1:
    alone = st.slider("🧘 Waktu Sendiri per Hari (jam)", 0, 20, 5)
    fear = st.slider("😰 Takut Bicara di Depan Umum (0=santai, 20=takut)", 0, 20, 5)
    event = st.slider("🎉 Kehadiran Acara Sosial", 0, 20, 7)

with col2:
    outside = st.slider("🌳 Sering Keluar Rumah?", 0, 20, 5)
    drained = st.slider("🥱 Lelah Setelah Sosialisasi?", 0, 20, 5)
    circle = st.slider("👥 Besarnya Lingkaran Pertemanan", 0, 20, 5)
    post = st.slider("📱 Frekuensi Posting di Media Sosial", 0, 20, 5)

# === Prediksi === #
if st.button("🔍 Prediksi Kepribadian Saya"):
    input_data = np.array([[alone, fear, event, outside, drained, circle, post]])
    prediction = model.predict(input_data)[0]

    st.markdown("---")
    if prediction == 0:
        st.success("🌙 **Kamu cenderung seorang Introvert!**")
        st.info("💬 *“Jalani hidup Anda seperti yang Anda inginkan, bukan seperti cara masyarakat memberi tahu Anda.”*")
        st.image("https://cdn-icons-png.flaticon.com/512/8090/8090401.png", width=120)
    else:
        st.success("🌞 **Kamu cenderung seorang Ekstrovert!**")
        st.info("💬 *“Terkadang menjadi ekstrovert memanglah sangat menguntungkan.”*")
        st.image("https://cdn-icons-png.flaticon.com/512/4111/4111123.png", width=120)

    st.markdown("## 🎯 Terima kasih telah mencoba! Jangan lupa bagikan hasilmu ✨")
