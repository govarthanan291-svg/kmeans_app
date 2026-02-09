import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="centered")

# ---------- Custom CSS ----------
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
    padding: 12px 25px;
    border-radius: 12px;
    border: none;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(to right, #24c6dc, #514a9d);
}

.result-box {
    background-color: #f0f8ff;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    text-align: center;
    font-size: 24px;
    color: green;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ---------- UI ----------
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("<h1>🛍️ Customer Segmentation App</h1>", unsafe_allow_html=True)
st.write("### Enter Customer Details")

income = st.number_input("💰 Annual Income (k$)", min_value=0, max_value=200, step=1)
score = st.number_input("⭐ Spending Score (1-100)", min_value=0, max_value=100, step=1)

cluster_labels = {
    0: "Low Spending Customer",
    1: "High Value Customer",
    2: "Average Customer",
    3: "Careful Customer",
    4: "Potential Customer"
}

if st.button("🔍 Predict Customer Stage"):
    user_data = np.array([[income, score]])
    user_scaled = scaler.transform(user_data)
    cluster = model.predict(user_scaled)[0]

    stage = cluster_labels.get(cluster, "Unknown Group")

    st.markdown(
        f"<div class='result-box'>Customer belongs to: <b>{stage}</b></div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
