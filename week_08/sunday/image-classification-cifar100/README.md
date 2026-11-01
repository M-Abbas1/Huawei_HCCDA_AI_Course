# CIFAR-100 Image Classification Project

Train TensorFlow/Keras image classifiers for the **100 fine-grained CIFAR-100 categories**, including a compact CNN and an ImageNet-pretrained MobileNetV2 transfer learning model. Serve the CNN with FastAPI and use it from a Streamlit app.

The training script loads CIFAR-100 from `datasets/CIFAR-100-dataset`. The dataset contains 50,000 training images and 10,000 test images. Images are 32 × 32 RGB pixels; the script reserves 5,000 training images for validation and keeps the test split for final evaluation.

## 1. Set up the environment

From this directory, create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow installation depends on operating system and hardware. Use the official [TensorFlow install guide](https://www.tensorflow.org/install) if the standard installation does not fit your machine.

## 2. Train the MobileNetV2 transfer learning model

```bash
python -m src.train --model mobilenetv2 --epochs 20
```

This loads pretrained ImageNet weights for MobileNetV2 and saves `models/cifar100_mobilenetv2.keras` and `models/mobilenetv2_metrics.json`. The first run may download the pretrained weights.

## 3. Train the CNN

```bash
python -m src.train --model cnn --epochs 30
```

The script uses the local dataset, creates a train/validation split, saves the best validation checkpoint, and evaluates it once on the separate test split. It saves:

- `models/cifar100_cnn.keras`
- `models/class_names.json`
- `models/cnn_metrics.json`

The models accept RGB pixel values in the range 0–255 at 32 × 32 resolution and rescale them internally. Training may take a while, especially on CPU; reduce `--epochs` for a quick smoke test.

CIFAR-100 is a challenging 100-class task. Accuracy depends on model capacity, training time, and hardware; the provided compact CNN is an educational baseline rather than a state-of-the-art model. Review the confusion matrix and per-class behavior before drawing conclusions from one overall accuracy score.

The previous Rock–Paper–Scissors model files are incompatible with this classifier. Train the CIFAR-100 CNN or MobileNetV2 before starting the API. The API loads `cifar100_mobilenetv2.keras` when present, otherwise it uses `cifar100_cnn.keras`.

## 4. Start the FastAPI service

Train a Keras model first. Then, in one terminal:

```bash
uvicorn api.main:app --reload
```

The service is available at `http://127.0.0.1:8000`. Interactive API docs are at `http://127.0.0.1:8000/docs`.

- `GET /health` checks that the service is running.
- `POST /predict` accepts one image upload and returns the predicted CIFAR-100 category, confidence, and scores for all 100 categories.

The API resizes uploaded images to 32 × 32, matching the model input shape. A model trained on CIFAR-100 may perform poorly on larger, differently framed real-world images.

## 5. Start the Streamlit app

In a second terminal, with the virtual environment active:

```bash
streamlit run app/streamlit_app.py
```

Upload an image to see the predicted class and the five highest class scores. The FastAPI service must also be running. To use a different API address, set `CIFAR100_API_URL` before starting Streamlit.

## Project layout

```text
rps_classifier/
├── api/main.py             # CIFAR-100 inference endpoint
├── app/streamlit_app.py    # Streamlit UI
├── src/data.py             # CIFAR-100 loading and preprocessing
├── src/model.py            # CNN and MobileNetV2 transfer learning definitions
├── src/train.py            # training, evaluation, and saved artifacts
└── models/                 # generated model files (not committed)
```

## Notes

- The dataset labels use CIFAR-100's 100 fine categories, in the order recorded in `models/class_names.json`.
- The official test split is not used for training or model selection.
- Model confidence is not a guarantee of correctness, especially for images outside CIFAR-100's image style and categories.
