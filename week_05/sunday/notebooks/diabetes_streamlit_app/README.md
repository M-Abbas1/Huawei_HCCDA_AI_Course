# Diabetes Prediction — Streamlit

## Files
- `app.py`: Streamlit interface
- `diabetes_model.pkl`: saved model bundle created from your notebook (add this file here)
- `requirements.txt`: required Python packages

## 1. Save the model
Run the model-saving cell at the end of your notebook. It creates `diabetes_model.pkl`.
Copy that file into this folder, next to `app.py`.

## 2. Install dependencies
Open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
```

## 3. Run the app

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, which you can open in your browser.

## Important
The feature names and their order must match the dataset used to train the model.
This demo is for educational purposes and is not a medical diagnostic tool.
