"""
computing classification criterion including per-class accuracy, OA, AA, and Kappa using different features
"""
from __future__ import print_function
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from EMO_ELM.classes.ELM import BaseELM
from sklearn.model_selection import StratifiedKFold, cross_val_score
from EMO_ELM.classes.ELM_nonlinear_RP import NRP_ELM
from Toolbox.Preprocessing import Processor

p = Processor()
path_1 = 'F:\Python\EMO_ELM\demo\experimental_results\X_projection\X_proj-Indian_pines_corrected-nh=10-iter=5000.npz'
path_2 = 'F:\Python\EMO_ELM\demo\experimental_results\Sparsity-dim-X_proj\Indian_pines-sparsity.npz'
npz = np.load(path_1)
X = np.load(path_2)['X_proj'][2]
y = npz['y']
time = npz['time']
print(time.tolist())
# ------------
# remove samples whose number is very small for IndianPines
# ------------
for c in np.unique(y):
    if np.nonzero(y == c)[0].shape[0] < 250:
        y = np.delete(y, np.nonzero(y == c))
y = p.standardize_label(y)

"""
=================
following lines add new random mapping
=================
"""
"""
# # add some data
root = 'F:\\Python\\HSI_Files\\'
im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
# im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# im_, gt_ = 'KSC', 'KSC_gt'

img_path = root + im_ + '.mat'
gt_path = root + gt_ + '.mat'

print(img_path)
img, gt = p.prepare_data(img_path, gt_path)
n_row, n_clo, n_bands = img.shape
print ('img=', img.shape)
# pca_img = p.pca_transform(n_comp, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, n_comp)
X_, y_ = p.get_correct(img, gt)
X_ = MinMaxScaler().fit_transform(X_)
print(X_.shape)
XX = NRP_ELM(10).fit(X_).predict(X_)
X[0, :, :] = XX
np.savez('./experimental_results/X_projection/X_proj-SalinasA_corrected-nh=10-iter=5000.npz', X_proj=X, y=npz['y'], time=npz['time'], score=npz['score'])
"""

# train_index, test_index = p.get_tr_tx_index(y, test_size=0.4)
# baseline_names = ['RP', 'PCA', 'SPCA', 'NMF', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE']
baseline_names = ['NRP', 'SPCA', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM($f_1$)', 'EMO-ELM($f_2$)', 'EMO-ELM(best)']

# for X_ in X:
#     X_train, X_test = X_[train_index], X_[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     elm = BaseELM(500, C=1e6)
#     y_predicted = elm.fit(X_train, y_train).predict(X_test)
#     ca, oa, aa, kappa = p.score(y_test, y_predicted)
#     ca = (np.asarray(ca) * 100).round(2)
#     oa = (np.asarray(oa) * 100).round(2)
#     aa = (np.asarray(aa) * 100).round(2)
#     kappa = np.asarray(kappa).round(3)
#     print(ca, '  \n',  oa, '  ',   aa,  '  ',   kappa)

all_aa, all_oa, all_kappa, all_ca = [], [], [], []
skf = StratifiedKFold(n_splits=3, random_state=55, shuffle=True)
for j in range(X.shape[0]):
    X_ = X[j]
    y_pres = []
    y_tests = []
    for i in range(10):
        for train_index, test_index in skf.split(X_, y):  # [index]
            X_train, X_test = X_[train_index], X_[test_index]  # [index]
            y_train, y_test = y[train_index], y[test_index]
            elm = BaseELM(500, C=1e8)
            y_predicted = elm.fit(X_train, y_train).predict(X_test)
            y_pres.append(y_predicted)
            y_tests.append(y_test)
    ca, oa, aa, kappa = p.save_res_4kfolds_cv(np.asarray(y_pres), np.asarray(y_tests), file_name=None, verbose=True)
    all_oa.append(oa)
    all_aa.append(aa)
    all_kappa.append(kappa)
    all_ca.append(ca)

np.savez('./experimental_results/Acc_comparison/IndianPines-Box_plot_data.npz', ca=all_ca, oa=all_oa, aa=all_aa, kappa=all_kappa)
print ('DONE')

