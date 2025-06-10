import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Judul aplikasi
st.title("🔮 Prediksi Kepribadian: Introvert atau Extrovert")

# Deskripsi singkat
st.markdown("""
Selamat datang di aplikasi prediksi kepribadian! 🎭  
Isi karakteristikmu di sisi kiri dan lihat apakah kamu lebih cenderung Introvert atau Extrovert.
""")

# Load model dan encoder
try:
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("❌ File model.pkl atau label_encoder.pkl tidak ditemukan. Pastikan file sudah diunggah.")
    st.stop()

# Fitur yang digunakan
features = [
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]

# Sidebar input
st.sidebar.header("📝 Masukkan Karakteristik Diri Kamu")
user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.slider(
        feature.replace("_", " "), min_value=0, max_value=20, value=5
    )

# Ubah input ke DataFrame
input_df = pd.DataFrame([user_input])

# Tampilkan input pengguna
st.subheader("📊 Data yang Kamu Masukkan")
st.write(input_df)

# Prediksi
prediction = model.predict(input_df)[0]
predicted_label = le.inverse_transform([prediction])[0]

# Tampilkan hasil prediksi
st.subheader("🧠 Hasil Prediksi Kepribadian")
if predicted_label == "Introvert":
    st.success("🌙 Kamu cenderung **Introvert**")
    st.markdown("> _Jalani hidup Anda seperti yang Anda inginkan, bukan seperti cara masyarakat memberi tahu Anda._")
elif predicted_label == "Extrovert":
    st.success("🌞 Kamu cenderung **Extrovert**")
    st.markdown("> _Terkadang menjadi ekstrovert memanglah sangat menguntungkan._")
else:
    st.warning("Hasil prediksi tidak dikenali.")
