from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import io


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Rock Paper Scissors API",
    description="ANN-based image classification API",
    version="1.0"
)


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

MODEL_PATH = "model/rock_paper_scissors_ann.keras"

model = load_model(MODEL_PATH)


# --------------------------------------------------
# 3. Load class names
# --------------------------------------------------

with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

print("Class names:", class_names)


# --------------------------------------------------
# 4. Preprocess image
# --------------------------------------------------

def preprocess_image(image):

    # Resize to the same size used during training
    image = image.resize((64, 64))

    # Convert to NumPy array
    image = np.array(image)

    # If image has an alpha channel, remove it
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Normalize pixels
    image = image / 255.0

    # Add batch dimension
    # (64, 64, 3)
    #       ↓
    # (1, 64, 64, 3)
    image = np.expand_dims(image, axis=0)

    return image


# --------------------------------------------------
# 5. Root endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Rock Paper Scissors ANN API is running!"
    }


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        contents = await file.read()

        # Convert bytes to PIL image
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Preprocess
        processed_image = preprocess_image(image)

        # Prediction
        probabilities = model.predict(
            processed_image,
            verbose=0
        )[0]

        # Get predicted class
        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = class_names[
            str(predicted_index)
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        # Return JSON response
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": {
                class_names[str(i)]: float(probabilities[i])
                for i in range(len(probabilities))
            }
        }

    except Exception as e:

        return JSONResponse(
            status_code=400,
            content={
                "error": str(e)
            }
        )