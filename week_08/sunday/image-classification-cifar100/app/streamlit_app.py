"""Streamlit front end for the FastAPI CIFAR-100 classifier."""

import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("CIFAR100_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="CIFAR-100 Image Classifier", page_icon="🖼️")
st.title("CIFAR-100 Image Classifier")
st.write("Upload an image and classify it into one of the 100 CIFAR-100 categories.")
st.caption(
    "CIFAR-100 images are 32 × 32 pixels; results on unrelated images may be unreliable."
)

uploaded = st.file_uploader("Choose an image", type=("png", "jpg", "jpeg", "webp"))
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    upload_key = (uploaded.name, len(image_bytes))
    if st.session_state.get("upload_key") != upload_key:
        st.session_state.pop("prediction", None)
        st.session_state["upload_key"] = upload_key

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify image", type="primary"):
        try:
            response = requests.post(
                f"{API_URL.rstrip('/')}/predict",
                files={
                    "file": (
                        uploaded.name,
                        image_bytes,
                        uploaded.type or "image/jpeg",
                    )
                },
                timeout=30,
            )
            response.raise_for_status()
            st.session_state["prediction"] = response.json()
        except requests.RequestException as exc:
            st.error(
                f"Could not reach the prediction API at {API_URL}. "
                f"Start FastAPI and try again. ({exc})"
            )
        except (KeyError, ValueError):
            st.error("The API returned an unexpected response.")

    result = st.session_state.get("prediction")
    if result:
        predicted_class = result["predicted_class"].replace("_", " ").title()
        st.subheader(f"Prediction: {predicted_class}")
        st.metric("Model confidence", f"{result['confidence']:.1%}")
        top_scores = sorted(
            result["scores"].items(), key=lambda item: item[1], reverse=True
        )[:5]
        st.write("Top 5 class scores")
        st.table(
            [
                {"Class": name.replace("_", " ").title(), "Score": f"{score:.1%}"}
                for name, score in top_scores
            ]
        )
