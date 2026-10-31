"""FastAPI endpoint for the trained Rock–Paper–Scissors CNN."""

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "rps_cnn.keras"
LABELS_PATH = ROOT / "models" / "class_names.json"
IMAGE_SIZE = (96, 96)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("CNN model files are missing. Run `python -m src.train --model cnn` first.")
    app.state.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    app.state.class_names = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    yield


app = FastAPI(title="Rock–Paper–Scissors CNN API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image file is larger than 10 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Upload a valid image file.") from None

    # The model includes its own Rescaling layer; send RGB pixel values in [0, 255].
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    scores = app.state.model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(scores))
    return {
        "predicted_class": app.state.class_names[predicted_index],
        "confidence": float(scores[predicted_index]),
        "scores": {
            name: float(scores[i]) for i, name in enumerate(app.state.class_names)
        },
    }
