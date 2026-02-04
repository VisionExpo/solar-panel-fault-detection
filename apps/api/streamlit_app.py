import time
from pathlib import Path
from typing import List

import requests
import streamlit as st
from PIL import Image

# ======================
# App Configuration
# ======================
st.set_page_config(
    page_title="Solar Panel Fault Detection",
    layout="centered",
)

st.title("☀️ Solar Panel Fault Detection")
st.write("Upload a solar panel image to detect faults using a trained ML model.")

# ======================
# Secrets & Config
# ======================
API_URL = st.secrets["API_URL"]
ENABLE_AUTO_REFRESH = st.secrets.get("ENABLE_AUTO_REFRESH", False)
REFRESH_INTERVAL = st.secrets.get("REFRESH_INTERVAL", 60)
MAX_BATCH_SIZE = st.secrets.get("MAX_BATCH_SIZE", 32)
ALLOWED_MIME_TYPES: List[str] = st.secrets.get(
    "ALLOWED_MIME_TYPES",
    ["image/jpeg", "image/jpg", "image/png"],
)

# ======================
# File Upload
# ======================
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    if uploaded_file.type not in ALLOWED_MIME_TYPES:
        st.error("Unsupported file type.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Inference"):
            with st.spinner("Running inference..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": uploaded_file.getvalue()},
                )

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction complete")

                st.write(f"**Predicted Class:** {result['predicted_class']}")
                st.write(f"**Confidence:** {result['confidence']:.2f}")

                with st.expander("Class Probabilities"):
                    st.json(result["probabilities"])
            else:
                st.error(f"API Error: {response.text}")

# ======================
# Auto Refresh
# ======================
if ENABLE_AUTO_REFRESH:
    time.sleep(REFRESH_INTERVAL)
    st.experimental_rerun()
