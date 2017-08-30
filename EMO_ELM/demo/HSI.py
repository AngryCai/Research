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

'''
__________________________________________________________________
Load HSI data
'''
gt_path = 'F:\Python\HSI_Files\SalinasA_gt.mat'
img_path = 'F:\Python\HSI_Files\SalinasA_corrected.mat'
#
# gt_path = 'F:\Python\HSI_Files\Indian_pines_gt.mat'
# img_path = 'F:\Python\HSI_Files\Indian_pines_corrected.mat'
#
# # gt_path = 'F:\Python\HSI_Files\Pavia_gt.mat'
# # img_path = 'F:\Python\HSI_Files\Pavia.mat'

print(img_path)
p = Processor()
img, gt = p.prepare_data(img_path, gt_path)
n_comp = 6
n_row, n_clo, n_bands = img.shape
# pca_img = p.pca_transform(n_comp, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, n_comp)
X, y = p.get_correct(img, gt)


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

############################
#      feature extraction
############################
'''
step 1: set common parameters
'''
n_hidden = 50
max_iter = 5000

'''
step 2: train models
'''
# sparse random projection (SRP)
# instance_rp = RP(n_hidden, sparse=True)
# X_projection_rp = instance_rp.fit(X).predict(X)
start = time.clock()
instance_rp = random_projection.SparseRandomProjection(n_components=n_hidden)
X_projection_rp = instance_rp.fit_transform(X)
time_rp = round(time.clock() - start, 3)

# PCA
start = time.clock()
instance_pca = PCA(n_components=n_hidden)
X_projection_pca = instance_pca.fit_transform(X)
time_pca = round(time.clock() - start, 3)

# SPCA
start = time.clock()
instance_spca = SparsePCA(n_components=n_hidden)
X_projection_spca = instance_spca.fit_transform(X)
time_spca = round(time.clock() - start, 3)

# NMF
start = time.clock()
instance_nmf = NMF(n_components=n_hidden, init='random', random_state=0)
X_projection_nmf = instance_nmf.fit_transform(X)
time_nmf = round(time.clock() - start, 3)

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
start = time.clock()
instance_emo_elm = EMO_AE_ELM(n_hidden, sparse_degree=0.05, max_iter=max_iter, n_pop=100)
X_projection_emo_elm = instance_emo_elm.fit(X, X).predict(X)
# instance_emo_elm.save_evo_result('EMO-ELM-AE-results.npz')
time_emo_elmae = round(time.clock() - start, 3)

'''
step 3: classification
'''
train_index, test_index = p.get_tr_tx_index(y, test_size=0.4)
y_train, y_test = y[train_index], y[test_index]
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

time_list = [time_rp, time_pca, time_spca, time_nmf, time_elmae, time_selmae, time_ae, time_sae, time_emo_elmae]
# NMSE_list = [Helper.calculate_NMSE(X, X_) for X_ in X_projection_list]
print ('------------------------------------')
print ('time:', time_list)
print ('------------------------------------')
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=1e4),
    DecisionTreeClassifier(max_depth=5),
    BaseELM(100, C=1e5),
    GaussianNB()]

baseline_names = ['RP', 'PCA', 'SPCA', 'NMF', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE']
classifier_names = ['KNN', 'SVM', 'DT', 'ELM', 'NB']
# results = {}
results = []
for i in range(X_projection_list.__len__()):
    print('---------------------------------')
    print('baseline: ', baseline_names[i])
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
np.savez('./experimental_results/X_projection.npz', X_proj=np.array(X_projection_list), time=time_list, y=y, score=results)

