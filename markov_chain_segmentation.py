import numpy as np
import matplotlib as plt
from tools import bruit_gauss, calc_erreur
from markov_chain import *


####Code question Q.4
taille_chaine = 1000
w = [0,1]
p = [0.25, 0.75]
A=np.array([[0.2,0.8],[0.8,0.2]])
m1 = 0
sig1 = 1
m2 = 3
sig2 = 2

chaine_markov = simu_mc(taille_chaine, w, p, A)
#print(chaine_markov)

chaine_markov_bruitee = bruit_gauss(chaine_markov, w, m1, sig1, m2, sig2)
#print (chaine_markov_bruitee)

chaine_markov_segmente = mpm_mc(chaine_markov_bruitee, w, p, A, m1, sig1, m2, sig2)
#print(chaine_markov_segmente)

error = calc_erreur(chaine_markov_segmente, chaine_markov)
print("Taux d'erreur : {0}".format(error))


####Calcul Q.5
taille_chaine = 1000
w = [0,1]
p = [0.25, 0.75]
A=[np.array([[0.8,0.2],[0.2,0.8]]),np.array([[0.3,0.7],[0.7,0.3]]),np.array([[0.5,0.5],[0.5,0.5]])]
m1 = [0, 1, 0]
sig1 = [1, 1, 1]
m2 = [3, 1, 1]
sig2 = [2, 5, 1]

chaine_markov = []
for i in range(3):
    chaine_markov.append(simu_mc(taille_chaine, w, p, A[0]))

signaux_bruite = {}
signaux_segmente = {}
for s in range(3):
    signaux_bruite[s] = np.zeros((3, len(chaine_markov[s])))
    signaux_segmente[s] = np.zeros((3, len(chaine_markov[s])))
    for i in range(3):
        signaux_bruite[s][i] = bruit_gauss(chaine_markov[s], w, m1[i], sig1[i], m2[i], sig2[i])
        signaux_segmente[s][i] = mpm_mc(signaux_bruite[s][i], w, p, A[s], m1[i], sig1[i], m2[i], sig2[i])

for s in range(3):
    for i in range(3):
        error = calc_erreur(chaine_markov[s], signaux_segmente[s][i])
        print("Taux d'erreur signal {0} bruit {1} : {2}%".format(s + 1, i + 1, error * 100))