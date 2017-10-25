"""
Hyperspectral image classification
"""
from __future__ import print_function

import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from EMO_ELM.classes.ELM import BaseELM
from EMO_ELM.classes.EMO_AE_ELM import EMO_AE_ELM
from Toolbox.Preprocessing import Processor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

'''
__________________________________________________________________
Load HSI data
'''
gt_path = 'F:\Python\HSI_Files\SalinasA_gt.mat'
img_path = 'F:\Python\HSI_Files\SalinasA_corrected.mat'
#
# gt_path = 'F:\Python\HSI_Files\wuhanTM_gt.mat'
# img_path = 'F:\Python\HSI_Files\wuhanTM.mat'
#
# gt_path = 'F:\Python\HSI_Files\Indian_pines_gt.mat'
# img_path = 'F:\Python\HSI_Files\Indian_pines_corrected.mat'
#
# gt_path = 'F:\Python\HSI_Files\Pavia_gt.mat'
# img_path = 'F:\Python\HSI_Files\Pavia.mat'

# gt_path = 'F:\Python\HSI_Files\KSC_gt.mat'
# img_path = 'F:\Python\HSI_Files\KSC.mat'

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
n_hidden = 50
max_iter = 1000

train_index, test_index = p.get_tr_tx_index(y, test_size=0.4)
results = []
X_proj = []
for rho in np.arange(0.1, 1, 0.1):
    start = time.clock()
    instance_emo_elm = EMO_AE_ELM(n_hidden, sparse_degree=rho, max_iter=max_iter, n_pop=100)
    X_projection_emo_elm = instance_emo_elm.fit(X, X).predict(X)
    # instance_emo_elm.save_evo_result('./experimental_results/EMO-ELM-AE-results-KSC-50hidden.npz')
    time_emo_elmae = round(time.clock() - start, 3)
    X_proj.append(X_projection_emo_elm)
    # TODO: calculate accuracy
    X_train, X_test = X_projection_emo_elm[train_index], X_projection_emo_elm[test_index]  # [index]
    y_train, y_test = y[train_index], y[test_index]
    elm = BaseELM(500, C=1e8)
    y_predicted = elm.fit(X_train, y_train).predict(X_test)
    acc = accuracy_score(y_test, y_predicted)
    acc_ = round(acc * 100, 2)
    results.append(acc_)
    print ('rho:', rho, ' acc:', acc_)
np.savez('F:\Python\EMO_ELM\demo\experimental_results\SalinasA-1000iter-50hidden-sparsity_acc_X_proj_differ-rho.npz',
         X=np.asarray(X_proj), acc=np.asarray(results))
