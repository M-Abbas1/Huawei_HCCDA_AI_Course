from fastapi import FastAPI
import json

def load_patients():
    with open("patients.json", 'r') as f:
        data = json.load(f)
    return data



app = FastAPI()


@app.get("/")
def hello():
    return {"msg":"welcome from api"}


@app.get("/view")
def get_patients():
    return load_patients()


@app.get("/search/{patient_id}")
def search(patient_id):
    data = load_patients()
    if patient_id in data.keys():
        return data[patient_id]
    return {'error': f"patinet not found with id {patient_id}"}