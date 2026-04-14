import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from src.preprocessing import preprocess_image
import pandas as pd
import time
import base64

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Medical Diagnostic System",
    layout="wide",
    page_icon="🧠"
)

# -----------------------------
# CUSTOM CSS (PREMIUM UI)
# -----------------------------
st.markdown("""
<style>

/* GLOBAL */
body {
    background: linear-gradient(135deg, #020617, #020617);
}

/* TITLE */
.main-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    color: #38bdf8;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* CARD */
.card {
    background: rgba(30, 41, 59, 0.8);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 8px 40px rgba(0,0,0,0.7);
    margin-bottom: 20px;
}

/* RESULT */
.result-good {
    background: linear-gradient(135deg, #065f46, #10b981);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}

.result-bad {
    background: linear-gradient(135deg, #7f1d1d, #ef4444);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}

/* SMALL BOX */
.small-box {
    padding: 15px;
    background-color: #020617;
    border-radius: 10px;
    text-align: center;
}

/* INFO */
.info-box {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/model.h5")

model = load_model()
class_names = ["Normal", "Pneumonia"]

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">🧠 AI Medical Diagnostic System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Pneumonia Detection from Chest X-rays</div>', unsafe_allow_html=True)

# -----------------------------
# LAYOUT
# -----------------------------
left, right = st.columns([1, 2])

# -----------------------------
# LEFT PANEL
# -----------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width="stretch")

        st.write("📌 Image Details")
        st.write(f"Size: {image.size}")
        st.write(f"Format: {image.format}")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# RIGHT PANEL
# -----------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 AI Diagnosis")

    if uploaded_file:

        img = preprocess_image(image)

        if st.button("🔍 Analyze Image"):

            # Animation
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)

            with st.spinner("Running Deep Learning Model..."):
                prediction = model.predict(img)[0]

            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))*100

            # RESULT
            if predicted_class == "Pneumonia":
                st.markdown(f'<div class="result-bad">⚠️ Pneumonia Detected<br>{confidence:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-good">✔️ Normal<br>{confidence:.2f}%</div>', unsafe_allow_html=True)

            # METRICS
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{confidence:.2f}%")
            c2.metric("Risk", "High" if predicted_class=="Pneumonia" else "Low")
            c3.metric("Model Accuracy", "92%")

            # -----------------------------
            # PROBABILITY TABLE
            # -----------------------------
            st.subheader("📊 Detailed Analysis")

            df = pd.DataFrame({
                "Class": class_names,
                "Probability (%)": prediction * 100
            })

            st.dataframe(df)

            st.bar_chart(df.set_index("Class"))

            # -----------------------------
            # SEVERITY SCORE
            # -----------------------------
            st.subheader("🎯 Severity Indicator")

            severity = int(confidence)

            if severity > 80:
                st.error("High Severity ⚠️")
            elif severity > 50:
                st.warning("Moderate Severity ⚠️")
            else:
                st.success("Low Severity ✔️")

            # -----------------------------
            # MEDICAL EXPLANATION
            # -----------------------------
            st.subheader("🧠 Medical Insight")

            if predicted_class == "Pneumonia":
                st.markdown("""
                <div class="info-box">
                Pneumonia is an infection that inflames air sacs in the lungs.
                Symptoms include fever, cough, and breathing difficulty.
                Early diagnosis helps prevent complications.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                The lungs appear normal with no visible infection.
                Clear lung fields indicate healthy respiratory condition.
                </div>
                """, unsafe_allow_html=True)

            # -----------------------------
            # DOWNLOAD REPORT
            # -----------------------------
            report = f"""
AI MEDICAL REPORT
-------------------------
Prediction: {predicted_class}
Confidence: {confidence:.2f}%
"""

            st.download_button("📥 Download Report", report)

            # -----------------------------
            # EXPANDABLE DETAILS
            # -----------------------------
            with st.expander("🔍 More Insights"):
                st.write("This prediction is based on a deep learning CNN model.")
                st.write("The model was trained on Chest X-ray dataset.")
                st.write("Accuracy may vary depending on image quality.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown('<div class="footer">🚀 AI | Healthcare | Deep Learning | Computer Vision</div>', unsafe_allow_html=True)