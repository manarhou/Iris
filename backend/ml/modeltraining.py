from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Charger les données
iris = load_iris()

X = iris.data    # les 4 mesures # ['sepal length', 'sepal width', 'petal length', 'petal width']
Y = Y = iris.target_names[iris.target] # les espèces (0, 1, 2 au lieu de noms)  # ['setosa', 'versicolor', 'virginica']

# Découper en données d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Tester le modèle
score = model.score(X_test, Y_test)
print(f"Précision : {score * 100:.1f}%")  # pourcentage

joblib.dump(model, "/home/manar/Documents/Pipelinecicd/backend/model/modele_iris.pkl")
