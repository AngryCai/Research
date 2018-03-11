from __future__ import print_function
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from EMO_ELM.classes.ELM import BaseELM
from sklearn.model_selection import StratifiedKFold, cross_val_score
from EMO_ELM.classes.ELM_nonlinear_RP import NRP_ELM
from Toolbox.Preprocessing import Processor

p = Processor()
path = './experimental_results/Sparsity-dim-X_proj/KSC-sparsity.npz'
path_ = './experimental_results/X_projection/X_proj-KSC-nh=10-iter=5000.npz'

npz = np.load(path)
X_total = npz['X_proj']
y = np.load(path_)['y']

# ------------
# remove samples whose number is very small for IndianPines
# ------------

y = p.standardize_label(y)

baseline_names = ['NRP', 'SPCA', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM($f_1$)', 'EMO-ELM($f_2$)', 'EMO-ELM(best)']
oa_mean, oa_std = [], []
aa_mean, aa_std = [], []
kappa_mean, kappa_std = [], []

for dim in range(30):
    all_aa, all_oa, all_kappa, all_ca = [], [], [], []
    skf = StratifiedKFold(n_splits=3, random_state=55, shuffle=True)
    X_dim = X_total[dim]
    oa_mean_, oa_std_ = [], []
    aa_mean_, aa_std_ = [], []
    kappa_mean_, kappa_std_ = [], []
    for j in range(X_dim.shape[0]):
        X_ = X_dim[j]
        if X_ is None:
            oa_mean_.append(None), oa_std_.append(None)
            aa_mean_.append(None), aa_std_.append(None)
            kappa_mean_.append(None), kappa_std_.append(None)
            continue
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
        ca, oa, aa, kappa = p.save_res_4kfolds_cv(np.asarray(y_pres), np.asarray(y_tests), file_name=None,
                                                  verbose=False)
        aa_mean_.append(np.round(aa.mean(), 2)), aa_std_.append(np.round(aa.std(), 2))
        oa_mean_.append(np.round(oa.mean(), 2)), oa_std_.append(np.round(oa.std(), 2))
        kappa_mean_.append(np.round(kappa.mean(), 3)), kappa_std_.append(np.round(kappa.std(), 3))
    print('{0} dim is done.'.format(dim))
    aa_mean.append(aa_mean_), aa_std.append(aa_std_)
    oa_mean.append(oa_mean_), oa_std.append(oa_std_)
    kappa_mean.append(kappa_mean_), kappa_std.append(kappa_std_)
np.savez('./experimental_results/Acc_dim/KSC-dim-acc.npz', oa=(oa_mean, oa_std), aa=(aa_mean, aa_std), kappa=(kappa_mean, kappa_std))
print ('DONE')



'''
---------------------------------
sparsity measure 
---------------------------------
'''

# sparsity = []
# for dim in range(30):
#     X_dim = X_total[dim]
#     ss = []
#     for j in range(X_dim.shape[0]):
#         if X_dim[j] is None:
#             s = None
#         else:
#             s = np.sum(np.linalg.norm(X_dim[j], ord=2, axis=1)) / np.sum(np.linalg.norm(X_dim[j], ord=1, axis=1))
#         ss.append(s)
#     sparsity.append(ss)
# np.savez('F:\Python\EMO_ELM\demo\experimental_results\Sparsity-dim-X_proj\Spa-KSC-sum.npz', s=sparsity)
