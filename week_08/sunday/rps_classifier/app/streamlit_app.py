"""Streamlit front end for the FastAPI Rock–Paper–Scissors classifier."""

import os
import random

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("RPS_API_URL", "http://127.0.0.1:8000")
MOVES = ("rock", "paper", "scissors")

st.set_page_config(page_title="CNN Hand Gesture Classifier", page_icon="✊")
st.title("✊ Rock–Paper–Scissors CNN")
st.write("Upload a hand gesture and ask the CNN to classify it.")
st.caption("The model works best with one centered hand and a clear gesture.")

uploaded = st.file_uploader("Choose a hand image", type=("png", "jpg", "jpeg", "webp"))
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    upload_key = (uploaded.name, len(image_bytes))
    if st.session_state.get("upload_key") != upload_key:
        st.session_state.pop("prediction", None)
        st.session_state["upload_key"] = upload_key

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict gesture", type="primary"):
        try:
            response = requests.post(
                f"{API_URL.rstrip('/')}/predict",
                files={"file": (uploaded.name, image_bytes, uploaded.type or "image/jpeg")},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state["prediction"] = response.json()
        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API at {API_URL}. Start FastAPI and try again. ({exc})")
        except (KeyError, ValueError):
            st.error("The API returned an unexpected response.")

    result = st.session_state.get("prediction")
    if result:
        prediction = result["predicted_class"].lower()
        st.subheader(f"Prediction: {result['predicted_class'].title()}")
        st.metric("Model confidence", f"{result['confidence']:.1%}")
        st.write("Class scores")
        st.bar_chart(result["scores"])

        if prediction in MOVES and st.button("Play a round against the computer"):
            computer = random.choice(MOVES)
            st.write(f"You: **{prediction.title()}** · Computer: **{computer.title()}**")
            if prediction == computer:
                st.info("It's a tie.")
            elif (prediction, computer) in (("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")):
                st.success("You win!")
            else:
                st.warning("Computer wins this round.")
