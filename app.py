import streamlit as st
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# === Load Lottie Animation === #
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except:
        return None

# === Load trained model === #
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# === Page configuration === #
st.set_page_config(page_title="Personality Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>🧠 Personality Type Predictor</h1>", unsafe_allow_html=True)
st.write("")

# === Lottie Header === #
with st.container():
    col1, col2 = st.columns([2, 3])
    with col1:
        lottie = load_lottie_url("https://lottie.host/1a81391d-671e-47ab-9023-c5e5e27e4691/XcVPuulNHE.json")
        if lottie:
            st_lottie(lottie, height=230)
    with col2:
        st.markdown("### 🔍 Answer a few questions to discover your personality type!")
        st.write("This app helps you identify whether you lean towards being an **Introvert** or **Extrovert** based on your daily habits and social preferences.")

# === User Inputs === #
st.markdown("---")
st.markdown("## 📋 Please fill in the details below:")

col1, col2 = st.columns(2)
with col1:
    time_alone = st.slider("🕒 Hours spent alone daily", 0, 20, 5)
    stage_fear = st.slider("🎤 Fear of public speaking (0 = very confident, 20 = extremely fearful)", 0, 20, 10)
    social_events = st.slider("🎊 Attendance at social events (0 = never, 20 = very often)", 0, 20, 10)
with col2:
    going_outside = st.slider("🚶 Frequency of going outside (0 = rarely, 20 = daily)", 0, 20, 10)
    drained = st.slider("😫 Feeling drained after socializing (0 = never, 20 = always)", 0, 20, 10)
    friend_circle = st.slider("👫 Size of friend circle (0 = small, 20 = very large)", 0, 20, 10)
    post_freq = st.slider("📱 Frequency of social media posting", 0, 20, 10)

# === Prediction Output === #
if st.button("🔎 Predict My Personality"):
    user_input = np.array([[time_alone, stage_fear, social_events, going_outside, drained, friend_circle, post_freq]])
    prediction = model.predict(user_input)[0]

    st.markdown("---")
    if prediction == 1:
        st.success("🌙 **You are likely an Introvert!**")
        st.info("\"Live life the way you want to, not the way society tells you.\"")
        st.image("https://cdn-icons-png.flaticon.com/512/8090/8090401.png", width=120)
    else:
        st.success("🌞 **You are likely an Extrovert!**")
        st.info("\"Sometimes being an extrovert is a huge advantage.\"")
        st.image("https://cdn-icons-png.flaticon.com/512/4111/4111123.png", width=120)

    st.markdown("### 🎯 Thank you for using this app! Feel free to share your result.")
