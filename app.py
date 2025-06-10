import streamlit as st
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# === Load Lottie animation === #
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

# === App title & animation === #
st.set_page_config(page_title="Personality Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>🔮 Discover Your Personality 🔍</h1>", unsafe_allow_html=True)
st.write("")

# === Show animation and intro text === #
with st.container():
    col1, col2 = st.columns([2, 3])
    with col1:
        lottie = load_lottie_url("https://lottie.host/1a81391d-671e-47ab-9023-c5e5e27e4691/XcVPuulNHE.json")
        if lottie:
            st_lottie(lottie, height=230)
    with col2:
        st.markdown("### 📌 Fill in your habits and behavior:")
        st.write("Answer the questions below to see whether you're more of an **Introvert** or **Extrovert**! 🚀")

# === Input sliders === #
st.markdown("---")
st.markdown("## 🧠 Answer the questions:")

col1, col2 = st.columns(2)

with col1:
    alone = st.slider("🧘 Hours spent alone daily", 0, 20, 5)
    fear = st.slider("🎤 Stage fright level (0=confident, 20=very scared)", 0, 20, 5)
    event = st.slider("🎉 Frequency of attending social events", 0, 20, 7)

with col2:
    outside = st.slider("🌳 How often do you go outside?", 0, 20, 5)
    drained = st.slider("🥱 Tired after socializing? (0=not at all, 20=very tired)", 0, 20, 5)
    circle = st.slider("👥 Size of your friend circle", 0, 20, 5)
    post = st.slider("📱 Posting frequency on social media", 0, 20, 5)

# === Prediction === #
if st.button("🔍 Predict My Personality"):
    input_data = np.array([[alone, fear, event, outside, drained, circle, post]])
    prediction = model.predict(input_data)[0]

    st.markdown("---")
    if prediction == 1:
        st.success("🌙 You are likely an **Introvert**")
        st.info("💬 *\"Live life the way you want to, not how society tells you.\"*")
        st.image("https://cdn-icons-png.flaticon.com/512/8090/8090401.png", width=120)
    else:
        st.success("🌞 You are likely an **Extrovert**")
        st.info("💬 *\"Sometimes being an extrovert is truly an advantage.\"*")
        st.image("https://cdn-icons-png.flaticon.com/512/4111/4111123.png", width=120)

    st.markdown("## 🎯 Thanks for trying! Feel free to share your results ✨")
