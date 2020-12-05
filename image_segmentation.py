import numpy as np
import cv2 as cv
from tools import bruit_gauss,calc_erreur, peano_transform_img, transform_peano_in_img, line_transform_img, transform_line_in_img
from gaussian_mixture import *
from markov_chain import *
import matplotlib.pyplot as plt

w = [0,255]
p = [0.25, 0.75]
A=[np.array([[0.8,0.2],[0.2,0.8]]),np.array([[0.3,0.7],[0.7,0.3]]),np.array([[0.5,0.5],[0.5,0.5]])]
m1 = [0, 1, 0]
sig1 = [1, 1, 1]
m2 = [3, 1, 1]
sig2 = [2, 5, 1]

i = 2
iter = 10
img = cv.imread("images/zebre2.bmp", cv.IMREAD_GRAYSCALE)
img_signal = peano_transform_img(img)
img_bruite = bruit_gauss(img_signal, w, m1[i], sig1[i], m2[i], sig2[i])
p_est, A_est, m1_est, sig1_est, m2_est, sig2_est = estim_param_EM_mc(iter, img_bruite, p, A[i], m1[i], sig1[i], m2[i], sig2[i])
img_restaure_mc = mpm_mc(img_bruite, w, p_est, A_est, m1_est, sig1_est, m2_est, sig2_est)
p_est, m1_est, sig1_est, m2_est, sig2_est = estim_param_EM_gm(iter, img_bruite, p, m1[i], sig1[i], m2[i], sig2[i])
img_restaure_gm = mpm_gm(img_bruite, w, p_est, m1_est, sig1_est, m2_est, sig2_est)

error = calc_erreur(img_signal, img_restaure_mc)
print("Taux d'erreur signal restauré MC bruit {0} : {1}%".format(i + 1, error * 100))

error = calc_erreur(img_signal, img_restaure_gm)
print("Taux d'erreur signal restauré modèle indépendant bruit {0} : {1}%".format(i + 1, error * 100))

f, axarr = plt.subplots(1,4, figsize=(15,15)) 
axarr[0].imshow(img, cmap='gray')
axarr[1].imshow(transform_peano_in_img(img_bruite, 256), cmap='gray')
axarr[2].imshow(transform_peano_in_img(img_restaure_mc, 256), cmap='gray')
axarr[3].imshow(transform_peano_in_img(img_restaure_gm, 256), cmap='gray')
plt.show()