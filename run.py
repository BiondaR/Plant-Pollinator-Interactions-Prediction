#######################
# Trabalho 2          #
# Grafos e Aplicações #
# Bionda Rozin        #
#######################

# Imports
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import preprocessing as p
import numpy as np
import utils as u

X, Y = p.preprocessing()

log = open("log.txt", "w")

models = {
    "RandomForest": RandomForestClassifier(),
    "NaiveBayes": GaussianNB(),
    "MultilayerPerceptron": MLPClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

n_folds = 5

folds = u.fold_split(X, Y, n_folds)

accuracy = np.zeros(len(models))
brier_loss = np.zeros(len(models))

i = 0

for m in models:
    print(m, file=log)
    eval = u.fold_eval(models[m], X, Y, folds, n_folds)
    accuracy[i] = np.mean(eval[0])
    brier_loss[i] = np.mean(eval[1])
    print("acc:", accuracy[i], file=log)
    print("bl:", brier_loss[i], file=log)
    i+=1

u.plot_bar(accuracy, models.keys(), "Acurácia")
u.plot_bar(brier_loss, models.keys(), "Brier Score Loss")