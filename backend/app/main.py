from fastapi import FastAPI
import joblib

model = joblib.load("model/modele_iris.pkl")

app = FastAPI()

@app.get("/predict")
def prediction(x: float, y: float, z: float, w: float):
    result = model.predict([[x, y, z, w]])
    return {"prediction": result[0]}

    

