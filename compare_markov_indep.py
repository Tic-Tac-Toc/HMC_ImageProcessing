import numpy as np
import matplotlib as plt
from tools import bruit_gauss, calc_erreur
from gaussian_mixture import *
from markov_chain import *

n = 1000
w = [0,1]
p = [0.25, 0.75]
A=np.array([[0.8,0.2],[0.07,0.93]])
m1 = [0, 1, 0]
sig1 = [1, 1, 1]
m2 = [3, 1, 1]
sig2 = [2, 5, 1]

for i in range(3):

    chaine_markov_indep = simu_gm(n, w, p)
    signal_bruite = bruit_gauss(chaine_markov_indep, w, m1[i], sig1[i], m2[i], sig2[i])
    pEstime, AEstime = calc_probaprio_mc(chaine_markov_indep, w)
    signal_restore_indep = mpm_gm(signal_bruite, w, pEstime, m1[i], sig1[i], m2[i], sig2[i])
    signal_restore_mc = mpm_mc(signal_bruite, w, pEstime, AEstime, m1[i], sig1[i], m2[i], sig2[i])

    chaine_markov_mc = simu_mc(n, w, p, A)
    signal_bruite_mc = bruit_gauss(chaine_markov_mc, w, m1[i], sig1[i], m2[i], sig2[i])
    pEstime = calc_probaprio_gm(chaine_markov_mc, w)
    signal_restore_indep2 = mpm_gm(signal_bruite_mc, w, pEstime, m1[i], sig1[i], m2[i], sig2[i])
    signal_restore_mc2 = mpm_mc(signal_bruite_mc, w, pEstime, A, m1[i], sig1[i], m2[i], sig2[i])

    print("BRUIT {0} - Taux d'erreur signal indépendant - Base/Restauration via le modèle indépendant : {1}%".format(i+1, calc_erreur(signal_restore_indep, chaine_markov_indep)*100))
    print("BRUIT {0} - Taux d'erreur signal indépendant - Base/Restauration par MPM suivant la chaîne de markov : {1}%".format(i+1, calc_erreur(signal_restore_mc, chaine_markov_indep)*100))
    print("BRUIT {0} - Taux d'erreur signal mc - Base/Restauration via le modèle indépendant : {1}%".format(i+1, calc_erreur(signal_restore_indep2, chaine_markov_mc)*100))
    print("BRUIT {0} - Taux d'erreur signal mc - Base/Restauration par MPM suivant la chaîne de markov : {1}%".format(i+1, calc_erreur(signal_restore_mc2, chaine_markov_mc)*100))