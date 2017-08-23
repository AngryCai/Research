"""
This file test the sparsity of mapped data for
    'RP', 'PCA', 'SPCA', 'NMF', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE'.
"""
from __future__ import print_function

import numpy as np
from scipy.io import loadmat
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
from EMO_ELM.classes.RandomProjection import RP
from EMO_ELM.classes.SparseAE import SAE
from Toolbox.Preprocessing import Processor
from sklearn import random_projection
from sklearn.decomposition import NMF, SparsePCA
from EMO_ELM.classes.Helper import Helper

############################
#      prepare data
############################

'''
__________________________________________________________________
Load data sets
'''
#######   USPS
# path = 'F:\Python\UCIDataset-matlab\USPSTest.mat'
# path = 'F:\Matlab\ELM_TIP_codes\USPSTrain.mat'
# mat = loadmat(path)
# p = Processor()
# X, y = mat['P'].astype('float32'), mat['T'].reshape(-1).astype('int8')
# X = X.transpose()
# print('data set:', 'USPSTest', 'data size:', X.shape)


#######   UCI
path = 'F:\Python\UCIDataset-matlab\UCI_25.mat'
mat = loadmat(path)
keys = mat.keys()
keys.remove('__version__')
keys.remove('__header__')
keys.remove('__globals__')
keys.sort()
save_name = 'result.npz'
# # load data
# keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo']
p = Processor()
key = 'Iris'
data = mat[key]
X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
print('data set:', key, 'data size:', X.shape)

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

####################################
#     Loop of feature extraction
####################################

dims = range(2, 51, 2)
sparsity_outer = []
for dim in dims:

    '''
    step 1: set common parameters
    '''
    n_hidden = dim
    max_iter = 1000

    '''
    step 2: train models
    '''
    # PCA
    # instance_pca = PCA(n_components=n_hidden)
    # X_projection_pca = instance_pca.fit_transform(X)

    # ELM-AE
    instance_elm_ae = ELM_AE(n_hidden, activation='sigmoid', sparse=False)
    X_projection_elm_ae = instance_elm_ae.fit(X).predict(X)

    # SELM-AE
    instance_selm_ae = ELM_AE(n_hidden, activation='sigmoid', sparse=True)
    X_projection_selm_ae = instance_selm_ae.fit(X).predict(X)

    # EMO-ELM-AE
    instance_emo_elm = EMO_AE_ELM(n_hidden, sparse_degree=0.01, max_iter=max_iter, n_pop=100)
    X_projection_emo_elm = instance_emo_elm.fit(X, X).predict(X)

    X_projection_list = [
        # X_projection_pca,
        X_projection_elm_ae,
        X_projection_selm_ae,
        X_projection_emo_elm]
    '''
    step 3: compute sparsity for each X_proj
    '''
    sparsity_inner = []
    for X_ in X_projection_list:
        sparsity_inner.append(Helper.calculate_sparsity(X_))
    sparsity_outer.append(np.asarray(sparsity_inner))
    print ('dim:', dim, ' L2/L1:', sparsity_inner)

np.savez('./experimental_results/sparsity-USPSTest.npz', sparsity=np.asarray(sparsity_outer))
print ('process is done.')



####################################
#     pareto front
####################################
# n_hidden = 20
# max_iter = 10000
# instance_emo_elm = EMO_AE_ELM(n_hidden, sparse_degree=0.01, max_iter=max_iter, n_pop=100)
# X_projection_emo_elm = instance_emo_elm.fit(X, X).predict_linear(X)
# instance_emo_elm.save_evo_result('./experimental_results/EMO_results.npz')



