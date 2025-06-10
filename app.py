import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Daftar fitur yang digunakan
selected_features = [
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]

st.set_page_config(page_title="Prediksi Kepribadian", page_icon="🧠")
st.title("🧠 Prediksi Kepribadian: Introvert atau Extrovert")
st.markdown("Masukkan skor (0–10) untuk masing-masing aspek berikut:")

# Ambil input dari pengguna
user_input = {}
for feature in selected_features:
    user_input[feature] = st.slider(feature.replace("_", " "), 0, 10, 5)

# Konversi ke DataFrame
input_df = pd.DataFrame([user_input])

# Prediksi kepribadian
if st.button("Prediksi"):
    prediction = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi: **{prediction}**")
    st.info("Extrovert cenderung nyaman bersosialisasi, sedangkan Introvert lebih suka refleksi pribadi.")
