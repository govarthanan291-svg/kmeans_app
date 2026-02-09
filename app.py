import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="centered")

# Custom CSS
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

.main {
    background-color: #ffffff;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0px 0px 25px rgba(0,0,0,0.2);
}

h1 {
    text-align: center;
    color: #4B0082;
    font-size: 42px;
}

label {
    font-size: 18px !important;
    color: #333333 !important;
}

.stButton > button {
    background: linear-gradient(to right, #ff512f, #dd2476);
    color: white;
    font-size: 20px;
    padding: 10px 25px;
    border-radius: 12px;
    border: none;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(to right, #24c6dc, #514a9d);
    color: white;
}

.result-box {
    background-color: #f0f8ff;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    text-align: center;
    font-size: 24px;
    color: #008000;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# App container
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("<h1>🛍️ Customer Segmentation App</h1>", unsafe_allow_html=True)
st.write("### Enter Customer Details")

income = st.slider("💰 Annual Income (k$)", 0, 200, 50)
score = st.slider("⭐ Spending Score (1-100)", 0, 100, 50)

if st.button("🔍 Predict Cluster"):
    data = np.array([[income, score]])
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)[0]

    st.markdown(
        f"<div class='result-box'>Customer belongs to <b>Cluster {cluster}</b></div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
