# importations
import numpy as np

# générer un dataset aléatoire
np.random.seed(0)
x = np.random.rand(300, 1)
# declaration des parametres de l'equation ax+b
a = 100
b = 350
# ajout d'une erreur
residu = 2*np.random.rand(300, 1)-1
# calcul de y avec passage de b dans la matrice x qui devient bidimensionnelle
y = a * x + residu + b



# fonction pour l'entraînement de la régression linéaire
def entrainement_reg_lineaire(X, Y, learning_rate, epoches):
    m = len(X)  # Nombre d'exemples d'entraînement
    theta = np.random.rand(2, 1)  # Initialisation aléatoire des paramètres (pente et ordonnée à l'origine)

    for epoche in range(epoches):
        # Calcul de la prédiction
        Y_pred = theta[0] * X + theta [1]

        # Calcul de l'erreur
        error = Y_pred - Y

        # Mise à jour des paramètres de manière vectorisée
        gradientA = (1 / m) * X.T.dot(error)
        theta[0] = theta[0] - learning_rate * gradientA
        gradientB = (1 / m) * np.sum(error)
        theta[1] = theta[1] - learning_rate * gradientB

    return theta



# Entraînement du modèle
learning_rate = 0.01
epochs = 10000
theta_final = entrainement_reg_lineaire(x, y, learning_rate, epochs)
print("Equation réel : ",a,"x +",b,"\n")
print("Approximation : ",theta_final[0],"x +",theta_final[1])