import joblib

import os

# Chemin relatif depuis la racine du backend
model = joblib.load(os.path.join(os.path.dirname(__file__), "../model/modele_iris.pkl"))

def test_prediction_setosa():
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])[0]
    assert prediction == "setosa" 

def test_prediction_versicolor():
    prediction = model.predict([[5.2, 2.2, 3.0, 1.2]])[0]
    assert prediction == "versicolor"