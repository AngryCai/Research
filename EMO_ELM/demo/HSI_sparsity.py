"""
This file test the sparsity of mapped data for
    'RP', 'PCA', 'SPCA', 'NMF', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE'.
"""
from __future__ import print_function
# import sys
# sys.path.extend(['/home/cym/python_codes/'])

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
from EMO_ELM.classes.ELM_nonlinear_RP import NRP_ELM

############################
#      prepare data
############################


'''
__________________________________________________________________
Load HSI data
'''
root = 'F:\\Python\\HSI_Files\\'
im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
# im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# im_, gt_ = 'KSC', 'KSC_gt'

img_path = root + im_ + '.mat'
gt_path = root + gt_ + '.mat'


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

####################################
#     Loop of feature extraction
####################################
dims = range(10, 301, 10)
sparsity_outer = []
X_proj = []
evo_results = []
for dim in dims:

    '''
    step 1: set common parameters
    '''
    n_hidden = dim
    max_iter = 5000

    '''
    step 2: train models
    '''
    # Nonlinear random ELM
    instance_nrp_elm = NRP_ELM(n_hidden)
    X_projection_nrp_elm = instance_nrp_elm.fit(X).predict(X)

    # PCA
    if dim <= X.shape[1]:
        # SPCA
        instance_spca = SparsePCA(n_components=n_hidden)
        X_projection_spca = instance_spca.fit_transform(X)
    else:
        X_projection_spca = None

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
    instance_emo_elm = EMO_AE_ELM(n_hidden, sparse_degree=0.05, max_iter=max_iter, n_pop=50)
    X_projection_emo_elm_best = instance_emo_elm.fit(X, X).predict(X)  # default using knee-based decision

    # # min f1/f2 decision
    evo_results__ = instance_emo_elm.get_result()
    non_obj = evo_results__['non_obj']
    f1_index, f2_index = non_obj[:, 0].argmin(), non_obj[:, 1].argmin()
    X_projection_emo_elm_f1 = instance_emo_elm.predict(X, W=evo_results__['W'][f1_index])
    X_projection_emo_elm_f2 = instance_emo_elm.predict(X, W=evo_results__['W'][f2_index])
    evo_results.append(evo_results__)  # # save this loop results

    X_projection_list = [
        X_projection_nrp_elm,
        X_projection_spca,
        X_projection_elm_ae,
        X_projection_selm_ae,
        X_projection_ae,
        X_projection_sae,
        X_projection_emo_elm_f1,
        X_projection_emo_elm_f2,
        X_projection_emo_elm_best
    ]

    X_proj.append(X_projection_list)  # save X projection
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

np.savez('./experimental_results/sparsity.npz', sparsity=np.asarray(sparsity_outer), X_proj=X_proj, evo_results=evo_results)
print ('process is done.')








