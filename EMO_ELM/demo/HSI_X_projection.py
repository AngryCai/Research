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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from EMO_ELM.classes.Helper import Helper
from EMO_ELM.classes.Autoencoder import Autoencoder
from EMO_ELM.classes.ELM import BaseELM
from EMO_ELM.classes.ELM_AE import ELM_AE
from EMO_ELM.classes.EMO_AE_ELM import EMO_AE_ELM
from EMO_ELM.classes.SparseAE import SAE
from Toolbox.Preprocessing import Processor
from platypus.algorithms import NSGAII
from EMO_ELM.classes.ELM_nonlinear_RP import NRP_ELM

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

for c in classes:
    if np.nonzero(y == c)[0].shape[0] < 10:
        X = np.delete(X, np.nonzero(y == c), axis=0)
        y = np.delete(y, np.nonzero(y == c))
y = p.standardize_label(y)
X = MinMaxScaler().fit_transform(X)
print ('size:', X.shape, 'n_classes:', np.unique(y).shape[0])

############################
#      feature extraction
############################
'''
step 1: set common parameters
'''
n_hidden = 10
max_iter = 5000



'''
step 2: train models
'''
# sparse random projection (SRP)
# instance_rp = random_projection.SparseRandomProjection(n_hidden)
# X_projection_rp = instance_rp.fit_transform(X)

# Nonlinear random ELM
start = time.clock()
instance_nrp_elm = NRP_ELM(n_hidden)
X_projection_nrp_elm = instance_nrp_elm.fit(X).predict(X)
time_rp = round(time.clock() - start, 3)

# SPCA
start = time.clock()
instance_spca = SparsePCA(n_components=n_hidden)
X_projection_spca = instance_spca.fit_transform(X)
time_spca = round(time.clock() - start, 3)

# ELM-AE
start = time.clock()
instance_elm_ae = ELM_AE(n_hidden, activation='sigmoid', sparse=False)
X_projection_elm_ae = instance_elm_ae.fit(X).predict(X)
time_elmae = round(time.clock() - start, 3)

# SELM-AE
start = time.clock()
instance_selm_ae = ELM_AE(n_hidden, activation='sigmoid', sparse=True)
X_projection_selm_ae = instance_selm_ae.fit(X).predict(X)
time_selmae = round(time.clock() - start, 3)

# AE
start = time.clock()
instance_ae = Autoencoder(n_hidden, max_iter=max_iter)
X_projection_ae = instance_ae.fit(X).predict(X)
time_ae = round(time.clock() - start, 3)

# SAE
start = time.clock()
instance_sae = SAE(n_hidden, max_iter=max_iter)
X_projection_sae = instance_sae.fit(X).predict(X)
time_sae = round(time.clock() - start, 3)

# EMO-ELM
# start = time.clock()
# instance_emo_elm = EMO_AE_ELM(n_hidden, sparse_degree=0.2, max_iter=max_iter, n_pop=100)
# X_projection_emo_elm = instance_emo_elm.fit(X, X).predict(X)
# instance_emo_elm.save_evo_result('./experimental_results/EMO-ELM-AE-results-KSC-50hidden.npz')
# time_emo_elmae = round(time.clock() - start, 3)

"""
-------------------------
using pretrained parameters to generate ELM
"""
model_path = 'F:\Python\EMO_ELM\demo\experimental_results\evo_result\EVO_RES-Indian_pines_corrected-nh=10-iter=5000-rho=0.05-n_pop=50.npz'
res = np.load(model_path)
obj = res['non_obj']
W = res['W']
B = res['B']

# EMO-ELM f1
instance_emo_elm_f2 = NRP_ELM(n_hidden).fit(X, W=W[obj[:, 0].argmin()])
X_projection_emo_elm_f1 = instance_emo_elm_f2.predict(X)

# EMO-ELM f2
instance_emo_elm_f2 = NRP_ELM(n_hidden).fit(X, W=W[obj[:, 1].argmin()])
X_projection_emo_elm_f2 = instance_emo_elm_f2.predict(X)

# EMO-ELM best
cur, (fx, fy), index = Helper.curvature_splines(obj[:, 0], obj[:, 1], s=0.005, k=3)
instance_emo_elm_best = NRP_ELM(n_hidden).fit(X, W=W[index[0]])
X_projection_emo_elm_best = instance_emo_elm_best.predict(X)

'''
step 3: classification
'''
train_index, test_index = p.get_tr_tx_index(y, test_size=0.4)
y_train, y_test = y[train_index], y[test_index]
X_projection_list = [
    X_projection_nrp_elm,
    # X_projection_rp,  # TODO: add original X_pro
    # X_projection_pca,  # TODO: add original X_pro
    X_projection_spca,
    # X_projection_nmf,  # TODO: add original X_pro
    X_projection_elm_ae,
    X_projection_selm_ae,
    X_projection_ae,  # TODO: add original X_pro
    X_projection_sae,
    X_projection_emo_elm_f1,
    X_projection_emo_elm_f2,
    X_projection_emo_elm_best,
]

time_list = [time_rp,
             # time_pca,
             time_spca,
             # time_nmf,
             time_elmae,
             time_selmae,
             # time_ae,
             time_sae,
             # time_emo_elmae
             ]
# NMSE_list = [Helper.calculate_NMSE(X, X_) for X_ in X_projection_list]
print ('------------------------------------')
print ('time:', time_list)
print ('------------------------------------')
classifiers = [
    KNeighborsClassifier(3),
    #SVC(kernel="linear", C=1e4),
    LinearSVC(),
    DecisionTreeClassifier(max_depth=5),
    BaseELM(500, C=1e5),
    GaussianNB()]

# baseline_names = ['RP', 'PCA', 'SPCA', 'NMF', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE']
baseline_names = ['RP', 'SPCA', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO_ELM_f1', 'EMO_ELM_f2', 'EMO_ELM_best']
classifier_names = ['KNN', 'SVM', 'DT', 'ELM', 'NB']
# results = {}
results = []
for i in range(X_projection_list.__len__()):
    print('---------------------------------')
    # print('baseline: ', baseline_names[i])
    X_ = X_projection_list[i]
    result_temp = []
    X_train, X_test = X_[train_index], X_[test_index]
    for j in range(classifiers.__len__()):
        clf = classifiers[j]
        clf.fit(X_train, y_train)
        score = np.round(clf.score(X_test, y_test)*100, 2)
        print('classifier: ', classifier_names[j], ' score:', score)
        result_temp.append(score)
    # results[baseline_names[i]] = result_temp
    results.append(np.asarray(result_temp))
results = np.asarray(results)
print('-----------------------------------')
print (classifier_names)
print (results)
# for k in baseline_names:
#     print (k, '||', results[k])
# save mapped data

result_name = 'X_proj-' + im_ + '-nh=' + str(n_hidden) + '-iter=' + str(max_iter) + '.npz'

np.savez('./experimental_results/X_projection/' + result_name,
X_proj=np.array(X_projection_list), time=time_list, y=y, score=results)
