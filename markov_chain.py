import numpy as np
from tools import gauss
import math as m


def forward(A, p, gauss):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs forward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param gauss: matrice (longeur du signal * 2) qui correspond aux valeurs des deux densité gaussiennes pour chaque élément du signal bruité
    :return: une matrice de taille: (la longueur de la chaîne * nombre de classe), contenant tous les forward (de 1 à n)
    """
    alpha = np.zeros((len(gauss), len(A)))
    alpha[0][0] = p[0] * gauss[0][0]
    alpha[0][1] = p[1] * gauss[0][1]
    alpha[0] /= np.sum(alpha[0])
    for j in range(1, len(gauss)):
        for k in range(len(A)):
            alpha[j][0] += gauss[j][0] * alpha[j-1][k] * A[k][0]
            alpha[j][1] += gauss[j][1] * alpha[j-1][k] * A[k][1]
        alpha[j] = alpha[j] / np.sum(alpha[j])

    return alpha


def backward(A, gauss):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs backward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param gauss: matrice (longeur du signal * 2) qui correspond aux valeurs des deux densités gaussiennes pour chaque élément du signal bruité
    :return: une matrice de taille: (la longueur de la chaîne * nombre de classe), contenant tous les backward (de 1 à n).
    Attention, si on calcule les backward en partant de la fin de la chaine, je conseille quand même d'ordonner le vecteur backward du début à la fin
    """
    T = len(gauss)
    beta = np.zeros((T, len(A)))
    beta[T - 1][0], beta[T - 1][1] = 1, 1
    beta[T - 1] /= np.sum(beta[T - 1])
    
    for j in range (2, T):
        for i in range (len(A)):
            beta[T - j][0] += beta[T - j + 1][i] * A[0][i] * gauss[T - j + 1][i]
            beta[T - j][1] += beta[T - j + 1][i] * A[1][i] * gauss[T - j + 1][i]

        beta[T - j] = beta[T -j] / np.sum(beta[T - j])

    return beta


def mpm_mc(signal_noisy, w, p, A, m1, sig1, m2, sig2):
    """
     Cette fonction permet d'appliquer la méthode mpm pour retrouver notre signal d'origine à partir de sa version bruité et des paramètres du model.
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: L'écart type de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: L'écart type de la deuxième gaussienne
    :return: Un signal discret à 2 classe (numpy array 1D d'int), résultat de la segmentation par mpm du signal d'entrée
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    alpha = forward(A, p, gausses)
    beta = backward(A, gausses)

    gamma = alpha * beta

    X = []
    indexes = np.argmax(gamma, axis=1)
        
    for index in indexes:
        X.append(w[index])
        
    return np.array(X)

def calc_probaprio_mc(signal, w):
    """
    Cete fonction permet de calculer les probabilité a priori des classes w1 et w2 et les transitions a priori d'une classe à l'autre,
    en observant notre signal non bruité
    :param signal: Signal discret non bruité à deux classes (numpy array 1D d'int)
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :return: un vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe et une une matrice de taille (2*2), matrice de transition de la chaîne
    """
    p0 = 0
    for i in signal:
        if i == w[0]:
            p0 += 1
    p0 = p0/len(signal)
    p = np.array((p0, 1-p0))
    A = np.array([[0,0], [0,0]])

    sumW1 = 0
    sumW2 = 0

    for i in range(1, len(signal)):
        if signal[i-1] == w[0]:
            sumW1 += 1
            if signal[i] == w[0]:
                A[0][0] += 1
            else: 
                A[0][1] += 1
        else:
            sumW2 += 1
            if signal[i] == w[0]:
                A[1][0] += 1
            else:
                A[1][1] += 1
    A[0] = A[0]/sumW1
    A[1] = A[0]/sumW2
    return p, A


def simu_mc(n, w, p, A):
    """
    Cette fonction permet de simuler un signal discret à 2 classe de taille n à partir des probabilité d'apparition des deux classes et de la Matrice de transition
    :param n: taille du signal
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :return: Un signal discret à 2 classe (numpy array 1D d'int), signal généré par la chaîne de Markov dont les paramètres sont donnés en entrée
    """
    simu = np.zeros((n,), dtype=int)
    aux = np.random.multinomial(1, p)
    simu[0] = w[np.argmax(aux)]
    for i in range(1, n):
        aux = np.random.multinomial(1, A[np.where(w == simu[i - 1])[0][0], :])
        simu[i] = w[np.argmax(aux)]
    return simu


def calc_param_EM_mc(signal_noisy, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction permet de calculer les nouveaux paramètres estimé pour une itération de EM
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: L'écart type de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: L'écart type de la deuxième gaussienne
    :return: tous les paramètres réestimés donc p, A, m1, sig1, m2, sig2
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    alpha = forward(A, p, gausses)
    beta = backward(A, gausses)

    gamma = np.zeros((2,len(signal_noisy)))
    for t in range(len(signal_noisy) - 1):
        for i in range(2):
            for j in range (2):
                gamma[i][t] += calculate_pijn(i,j, t, alpha, beta, gausses, A)

    p_est = [gamma[0][0], gamma[1][0]]
    A_est = np.zeros((2,2))

    for i in range(2):
        for j in range(2):
            num, denom = 0, 0            
            for t in range(len(signal_noisy) -1):
                num += calculate_pijn(i,j, t, alpha, beta, gausses, A)
                denom += gamma[i][t]
            A_est[i][j] = num/denom
    
    num = [0, 0]
    denom = [0, 0]
    for t in range(len(signal_noisy)):
        print(str(t) +' 1\r', end="")
        num[0] += signal_noisy[t] * gamma[0][t]
        denom[0] += gamma[0][t]
        num[1] += signal_noisy[t] * gamma[1][t]
        denom[1] += gamma[1][t]
    m1_est = num[0]/denom[0]
    m2_est = num[1]/denom[1]

    num = [0, 0]
    denom = [0, 0]
    for t in range(len(signal_noisy)):
        print(str(t) +' 2\r', end="")
        num[0] += np.power(signal_noisy[t] - m1_est, 2) * gamma[0][t]
        denom[0] += gamma[0][t]
        num[1] += np.power(signal_noisy[t] - m2_est, 2) * gamma[1][t]
        denom[1] += gamma[1][t]
    sig1_est = np.sqrt(num[0]/denom[0])
    sig2_est = np.sqrt(num[1]/denom[1])

    return  p_est, A_est, m1_est, sig1_est, m2_est, sig2_est

def calculate_pijn(i, j, n, alpha, beta, gausses, A):
    print(str(n) +'\r', end="")
    numerateur = alpha[n][i] * A[i][j] * gausses[n + 1][j] * beta[n + 1][j]
    denom = 0
    for k in range(2):
        for l in range(2):
            denom += alpha[n][k] * A[k][l] * gausses[n + 1][l] * beta[n + 1][l]
    if (m.isnan(numerateur) or m.isnan(denom) or m.isnan(numerateur/denom)):
        print(i, j, n, numerateur, denom, numerateur/denom, alpha[n][k], A[k][l], gausses[n + 1][l], beta[n + 1][l])
        input()
    return numerateur / denom


def estim_param_EM_mc(iter, signal_noisy, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction est l'implémentation de l'algorithme EM pour le modèle en question
    :param iter: Nombre d'itération choisie
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: la valeur d'initialisation du vecteur de proba
    :param A: la valeur d'initialisation de la matrice de transition de la chaîne
    :param m1: la valeur d'initialisation de la moyenne de la première gaussienne
    :param sig1: la valeur d'initialisation de l'écart type de la première gaussienne
    :param m2: la valeur d'initialisation de la moyenne de la deuxième gaussienne
    :param sig2: la valeur d'initialisation de l'écart type de la deuxième gaussienne
    :return: Tous les paramètres réestimés à la fin de l'algorithme EM donc p, A, m1, sig1, m2, sig2
    """
    p_est = p
    A_est = A
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2
    for i in range(iter):
        p_est, A_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_EM_mc(signal_noisy, p_est, A_est, m1_est,
                                                                            sig1_est, m2_est, sig2_est)
        print({'p': p_est, 'A': A_est, 'm1': m1_est, 'sig1': sig1_est, 'm2': m2_est, 'sig2': sig2_est})
    return p_est, A_est, m1_est, sig1_est, m2_est, sig2_est
