"""
Hyperspectral image classification
"""
from __future__ import print_function

import time

import numpy as np
from sklearn import random_projection
from sklearn.decomposition import NMF, SparsePCA
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from EMO_ELM.classes.Autoencoder import Autoencoder
from EMO_ELM.classes.ELM import BaseELM
from EMO_ELM.classes.ELM_AE import ELM_AE
from EMO_ELM.classes.EMO_AE_ELM import EMO_AE_ELM
from EMO_ELM.classes.SparseAE import SAE
from Toolbox.Preprocessing import Processor
from platypus.algorithms import NSGAII

'''
__________________________________________________________________
Load HSI data
'''
root = 'F:\\Python\\HSI_Files\\'
# im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# im_, gt_ = 'KSC', 'KSC_gt'

img_path = root + im_ + '.mat'
gt_path = root + gt_ + '.mat'
#
# gt_path = 'F:\Python\HSI_Files\Indian_pines_gt.mat'
# img_path = 'F:\Python\HSI_Files\Indian_pines_corrected.mat'
#
# gt_path = 'F:\Python\HSI_Files\Pavia_gt.mat'
# img_path = 'F:\Python\HSI_Files\Pavia.mat'

print(img_path)
p = Processor()
img, gt = p.prepare_data(img_path, gt_path)
n_row, n_clo, n_bands = img.shape
print ('img=', img.shape)
# pca_img = p.pca_transform(n_comp, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, n_comp)
X, y = p.get_correct(img, gt)
print(X.shape)

''' 
___________________________________________________________________
Data pre-processing
'''
# remove these samples with small numbers
classes = np.unique(y)
print ('size:', X.shape, 'n_classes:', classes.shape[0])

for c in classes:
    if np.nonzero(y == c)[0].shape[0] < 10:
        X = np.delete(X, np.nonzero(y == c), axis=0)
        y = np.delete(y, np.nonzero(y == c))
y = p.standardize_label(y)
X = MinMaxScaler().fit_transform(X)

n_hidden = 10
max_iter = 5000
sparse_degree = 0.05
n_pop = 50
start = time.clock()
instance_emo_elm = EMO_AE_ELM(n_hidden, sparse_degree=sparse_degree, max_iter=max_iter, n_pop=n_pop)
X_projection_emo_elm = instance_emo_elm.fit(X, X).predict(X)

result_name = im_ + '-nh=' + str(n_hidden) + '-iter=' + str(max_iter) \
              + '-rho=' + str(sparse_degree) + '-n_pop=' + str(n_pop)

instance_emo_elm.save_evo_result('./experimental_results/evo_result/' + 'EVO_RES-' + result_name + '.npz')
time_emo_elmae = round(time.clock() - start, 3)
print ('TIME=', time_emo_elmae)
