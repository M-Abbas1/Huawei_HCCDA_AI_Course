"""FastAPI endpoint for a trained CIFAR-100 Keras model."""

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from src.model import build_cnn, build_mobilenetv2

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    ROOT / "models" / "cifar100_mobilenetv2.keras",
    ROOT / "models" / "cifar100_cnn.keras",
)
LABELS_PATH = ROOT / "models" / "class_names.json"
IMAGE_SIZE = (32, 32)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = next((path for path in MODEL_PATHS if path.exists()), None)
    if not LABELS_PATH.exists():
        raise RuntimeError(f"Class labels file is missing: {LABELS_PATH}")
    if model_path is None:
        raise RuntimeError(
            "No supported Keras model found in the models folder. "
            "Train one with `python -m src.train --model mobilenetv2` or "
            "`python -m src.train --model cnn`."
        )
    app.state.class_names = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    try:
        app.state.model = tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError):
        # A .keras archive can contain weights even when its saved config uses
        # fields unsupported by the local Keras version. Rebuild our known model
        # definition, then load the archive's weights by layer topology.
        builders = {
            "cifar100_mobilenetv2.keras": lambda count: build_mobilenetv2(
                num_classes=count, weights=None
            ),
            "cifar100_cnn.keras": lambda count: build_cnn(num_classes=count),
        }
        builder = builders.get(model_path.name)
        if builder is None:
            raise
        app.state.model = builder(len(app.state.class_names))
        try:
            app.state.model.load_weights(model_path)
        except (TypeError, ValueError) as weights_error:
            raise RuntimeError(
                f"Could not load model {model_path.name}. Its architecture or Keras "
                "version may not match this project. Upgrade TensorFlow/Keras to the "
                "version used to save it, or export a compatible model."
            ) from weights_error
        print(f"Loaded weights from {model_path.name} using the local model definition")
    app.state.model_path = model_path
    print(f"Loaded model: {model_path.name}")
    yield


app = FastAPI(title="CIFAR-100 Image Classifier API", version="1.0.0", lifespan=lifespan)


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

    # Both supported models include their own input preprocessing.
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
