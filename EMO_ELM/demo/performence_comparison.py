"""
This file compare random projection(RP), ELM-AE, SELM-AE, AE, PCA and EMO-ELM in terms of sparsity and
effectiveness of feature extraction.
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
key = 'Iris'
data = mat[key]
X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')

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
n_hidden = 2
max_iter = 2000

'''
step 2: train models
'''
# random projection
instance_rp = RP(n_hidden)
X_projection_rp = instance_rp.fit(X).predict(X)

# PCA
instance_pca = PCA(n_components=n_hidden)
X_projection_pca = instance_pca.fit_transform(X)

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
X_projection_emo_elm = instance_emo_elm.fit(X, X).predict(X)
# instance_emo_elm.save_evo_result('EMO-ELM-AE-results.npz')

'''
step 3: classification
'''
train_index, test_index = p.get_tr_tx_index(y, test_size=0.4)
y_train, y_test = y[train_index], y[test_index]
X_projection_list = [
    X_projection_rp,
    X_projection_pca,
    X_projection_elm_ae,
    X_projection_selm_ae,
    X_projection_ae,
    X_projection_sae,
    X_projection_emo_elm]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=10000),
    DecisionTreeClassifier(max_depth=5),
    BaseELM(50),
    GaussianNB()]

baseline_names = ['RP', 'PCA', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE']
classifier_names = ['KNN', 'SVM', 'DT', 'ELM', 'NB']
results = {}
for i in range(X_projection_list.__len__()):
    print('---------------------------------')
    print('baseline: ', baseline_names[i])
    X_ = X_projection_list[i]
    result_temp = []
    X_train, X_test = X_[train_index], X_[test_index]
    for j in range(classifiers.__len__()):
        clf = classifiers[j]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('classifier: ', classifier_names[j], ' score:', score)
        result_temp.append(score)
    results[baseline_names[i]] = result_temp

print('-----------------------------------')
print (classifier_names)
for k in baseline_names:
    print (k, '||', results[k])
# save mapped data
np.savez('./experimental_results/X_projection.npz', X_proj=np.array(X_projection_list), y=y)
