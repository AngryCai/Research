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
Load UCI data sets
'''
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
key = 'wdbc'
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
dims = range(5, 51, 5)
sparsity_outer = []
for dim in dims:

    '''
    step 1: set common parameters
    '''
    n_hidden = dim
    max_iter = 2000

    '''
    step 2: train models
    '''
    # random projection
    # instance_rp = RP(n_hidden, sparse=True)
    # X_projection_rp = instance_rp.fit(X).predict(X)
    instance_rp = random_projection.SparseRandomProjection(n_components=n_hidden)
    X_projection_rp = instance_rp.fit_transform(X)

    # PCA
    if dim <= X.shape[1]:
        instance_pca = PCA(n_components=n_hidden)
        X_projection_pca = instance_pca.fit_transform(X)

        # SPCA
        instance_spca = SparsePCA(n_components=n_hidden)
        X_projection_spca = instance_spca.fit_transform(X)

        # NMF
        instance_nmf = NMF(n_components=n_hidden, init='random', random_state=0)
        X_projection_nmf = instance_nmf.fit_transform(X)
    else:
        X_projection_pca = None
        X_projection_spca = None
        X_projection_nmf = None

    # ELM-AE
    instance_elm_ae = ELM_AE(n_hidden, activation='sigmoid', sparse=False)
    X_projection_elm_ae = instance_elm_ae.fit(X).predict(X)

    # SELM-AE
    instance_selm_ae = ELM_AE(n_hidden, activation='sigmoid', sparse=True)
    X_projection_selm_ae = instance_selm_ae.fit(X).predict(X)

    # AE
    instance_ae = Autoencoder(n_hidden, max_iter=max_iter)
    X_projection_ae = instance_ae.fit(X).predict(X)

    # SAE
    instance_sae = SAE(n_hidden, max_iter=max_iter)
    X_projection_sae = instance_sae.fit(X).predict(X)

    # EMO-ELM
    instance_emo_elm = EMO_AE_ELM(n_hidden, sparse_degree=0.05, max_iter=max_iter, n_pop=100)
    X_projection_emo_elm = instance_emo_elm.fit(X, X).predict(X)
    # instance_emo_elm.save_evo_result('EMO-ELM-AE-results.npz')

    X_projection_list = [
        X_projection_rp,
        X_projection_pca,
        X_projection_spca,
        X_projection_nmf,
        X_projection_elm_ae,
        X_projection_selm_ae,
        X_projection_ae,
        X_projection_sae,
        X_projection_emo_elm]
    '''
    step 3: compute sparsity for each X_proj
    '''
    sparsity_inner = []
    for X_ in X_projection_list:
        if X_ is not None:
            sparsity_inner.append(Helper.calculate_sparsity(X_))  # Helper.calculate_sparsity(X_)
        else:
            sparsity_inner.append(None)
    sparsity_outer.append(np.asarray(sparsity_inner))
    print ('dim:', dim, ' L2/L1:', sparsity_inner)

np.savez('./experimental_results/sparsity.npz', sparsity=np.asarray(sparsity_outer))
print ('process is done.')








