# Rock–Paper–Scissors CNN Project

Train a TensorFlow/Keras image classifier for **rock**, **paper**, and **scissors**, compare it with a dense ANN baseline, then serve the CNN with FastAPI and use it from a Streamlit app.

Use the [Rock–Paper–Scissors image dataset](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors). Download and extract it yourself before class; this project does not download the dataset. The code expects the training and test folders shown below. It creates a fixed training/validation split from the training images and reserves the test folder for final evaluation.

## 1. Set up the environment

From this directory, create and activate a virtual environment, then install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow installation depends on operating system and hardware. Use the official [TensorFlow install guide](https://www.tensorflow.org/install) if the standard installation does not fit your machine.

## 2. Prepare the dataset

Create this folder structure inside the project directory (`week08/sunday/rps_classifier/`):

```text
datasets/
├── rps/
│   ├── paper/
│   ├── rock/
│   └── scissors/
└── rps-test-set/
    ├── paper/
    ├── rock/
    └── scissors/
```

Put each image inside its matching class folder. Keep the separate test images in `rps-test-set/`; do not move them into the training folder.

## 3. Saturday notebook

For Saturday's guided lesson, open the [CNN lab notebook](../../saturday/code/saturday_cnn_lab.ipynb) in JupyterLab:

```bash
jupyter lab ../../saturday/code/saturday_cnn_lab.ipynb
```

Run the notebook cells in order to inspect the local data, build the CNN, train it, review errors, and save `models/rps_cnn.keras` for Sunday's project introduction. Start JupyterLab from this project directory so the notebook can import `src` and find `datasets/`.

## 4. Train the ANN baseline

```bash
python -m src.train --model ann --epochs 8
```

This saves `models/rps_ann.keras` and `models/ann_metrics.json`.

## 5. Train the CNN

```bash
python -m src.train --model cnn --epochs 12
```

The script trains with a fixed train/validation split, selects the best validation checkpoint, and evaluates it once on the separate test split. It saves:

- `models/rps_cnn.keras`
- `models/class_names.json`
- `models/cnn_metrics.json`

The CNN includes resizing-compatible input, rescaling, augmentation during training, convolution, batch normalization, pooling, and a softmax output. The saved model accepts RGB pixel values in the range 0–255 at 96 × 96 resolution; it performs rescaling internally.

For a faster classroom run, reduce `--epochs` (for example, `--epochs 3`). Compare ANN and CNN results on the same test split. A low score can reflect data variation, capture differences, preprocessing, or model choices; inspect examples and the confusion matrix before changing the architecture.

## 6. Start the FastAPI service

Train the CNN first. Then, in one terminal:

```bash
uvicorn api.main:app --reload
```

The service is available at `http://127.0.0.1:8000`. Interactive API docs are at `http://127.0.0.1:8000/docs`.

- `GET /health` checks that the service is running.
- `POST /predict` accepts one image upload and returns the predicted class, confidence, and scores for all classes.

## 7. Start the Streamlit app

In a second terminal, with the virtual environment active:

```bash
streamlit run app/streamlit_app.py
```

Upload a clear image with one hand gesture. The app calls the FastAPI endpoint and displays class scores. Both services must be running. To use a different API address, set `RPS_API_URL` before starting Streamlit.

## Project layout

```text
rps_classifier/
├── api/main.py             # FastAPI inference endpoint
├── app/streamlit_app.py    # Streamlit UI
├── src/data.py             # local folder loading and preprocessing
├── src/model.py            # ANN baseline and CNN definitions
├── src/train.py            # training, evaluation, and saved artifacts
├── models/                 # generated model files (not committed)
└── datasets/               # extracted image folders (not committed)
```

## Notes for learners

- The API and training pipeline resize images to 96 × 96 RGB. The model rescales pixel values internally.
- Use the same class order from `models/class_names.json` when interpreting predictions.
- Test accuracy is only meaningful if the test set is kept out of training and model selection.
- A high confidence score is not a guarantee that an unfamiliar or poorly framed image is classified correctly.
- The hand gesture game is a demo of the classifier output, not a separate trained model.
