## Project Structure


rock_paper_scissors/  \
│   \
├── model/  \
│   ├── rock_paper_scissors_ann.keras  \
│   └── class_names.json  \
│   \
├── api.py   \
├── streamlit_app.py   \
└── requirements.txt 



`pip install fastapi uvicorn python-multipart pillow tensorflow`

`uvicorn api:app --reload`

`streamlit run streamlit_app.py`





                     USER
                      │
                      │ Upload Image
                      ▼
             ┌─────────────────┐
             │                 │
             │    STREAMLIT    │
             │    FRONTEND     │
             │                 │
             └────────┬────────┘
                      │
                      │ HTTP POST
                      │ /predict
                      ▼
             ┌─────────────────┐
             │                 │
             │     FASTAPI     │
             │     BACKEND     │
             │                 │
             └────────┬────────┘
                      │
                      │ Image
                      ▼
             ┌─────────────────┐
             │                 │
             │   PREPROCESS    │
             │                 │
             │ Resize 64×64    │
             │ Normalize       │
             │ Add batch       │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │                 │
             │       ANN       │
             │                 │
             │  Flatten        │
             │  Dense 128      │
             │  Dense 64       │
             │  Dense 3        │
             │  Softmax        │
             │                 │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    PREDICTION   │
             │                 │
             │ Paper: 96.4%    │
             │ Rock:   2.1%    │
             │ Scissor: 1.5%   │
             └────────┬────────┘
                      │
                      │ JSON
                      ▼
             ┌─────────────────┐
             │    STREAMLIT    │
             │                 │
             │ Prediction      │
             │ Confidence      │
             │ Probabilities   │
             └─────────────────┘