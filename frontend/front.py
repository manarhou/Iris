import streamlit as st 
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")  # ← variable d'env

st.title("Quelle Iris")

st.subheader("Entre les mesures de la fleur :")

x = st.number_input("Longueur du sépale (cm)", min_value=0.0, max_value=10.0, value=5.1)
y  = st.number_input("Largeur du sépale (cm)",  min_value=0.0, max_value=10.0, value=3.5)
z = st.number_input("Longueur du pétale (cm)", min_value=0.0, max_value=10.0, value=1.4)
w  = st.number_input("Largeur du pétale (cm)",  min_value=0.0, max_value=10.0, value=0.2)

if st.button("Prédire"):
    response = requests.get(f"{API_URL}/predict", params={
    "x": x,
    "y": y,
    "z": z,
    "w": w
})
    st.write(response.json())  # on teste d'abord




