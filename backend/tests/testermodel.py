import joblib

# Charger le modèle
model = joblib.load("/home/manar/Documents/Pipelinecicd/backend/model/modele_iris.pkl")


nouvelle_fleur = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(nouvelle_fleur)[0] # → [0] = setosa
print(prediction)  


# tester une prédiction sur une nouvelle fleur
nouvelle_fleur = [[5.2, 2.2, 3.0,1.2]]  # les 4 mesures
prediction = model.predict(nouvelle_fleur)[0] # "versicolor"
print(prediction)  