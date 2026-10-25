import streamlit as st
import requests
from PIL import Image


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Rock Paper Scissors AI",
    page_icon="✊",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("✊ Rock Paper Scissors AI")

st.write(
    "Upload an image of a hand gesture and "
    "let the ANN predict Rock, Paper, or Scissors."
)


# --------------------------------------------------
# API URL
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Display image
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )


    # --------------------------------------------------
    # Predict button
    # --------------------------------------------------

    if st.button(
        "🔮 Predict",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            try:

                # Reset file pointer
                uploaded_file.seek(0)

                # Prepare image for API
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type
                    )
                }

                # Send POST request
                response = requests.post(
                    API_URL,
                    files=files
                )

                # Check response
                if response.status_code == 200:

                    result = response.json()

                    prediction = result["prediction"]
                    confidence = result["confidence"]

                    # ------------------------------------------
                    # Display result
                    # ------------------------------------------

                    st.success(
                        f"Prediction: {prediction.upper()}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2%}"
                    )

                    # ------------------------------------------
                    # Display probabilities
                    # ------------------------------------------

                    st.subheader(
                        "Class Probabilities"
                    )

                    probabilities = result[
                        "probabilities"
                    ]

                    for class_name, probability in probabilities.items():

                        st.write(
                            f"**{class_name.capitalize()}**"
                        )

                        st.progress(
                            probability
                        )

                else:

                    st.error(
                        f"API Error: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure api.py is running."
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )