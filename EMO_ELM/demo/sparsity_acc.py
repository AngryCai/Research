from __future__ import print_function
import numpy as np
from EMO_ELM.classes.ELM import BaseELM
from Toolbox.Preprocessing import Processor
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

p = Processor()
path = 'F:\Python\EMO_ELM\demo\experimental_results\KSC-1000iter-50hidden-sparsity_acc_X_proj_differ-rho.npz'
p2 = 'F:\Python\EMO_ELM\demo\experimental_results\KSC-X_projection-10hidden-5000iter.npz'
npz = np.load(path)
X = npz['X']
acc = npz['acc']
y = np.load(p2)['y']

all_aa, all_oa, all_kappa = [], [], []
skf = StratifiedKFold(n_splits=5, random_state=55, shuffle=True)
for j in range(X.shape[0]):
    X_ = X[j]
    y_pres = []
    y_tests = []
    for i in range(1):
        for train_index, test_index in skf.split(X_, y):  # [index]
            X_train, X_test = X_[train_index], X_[test_index]  # [index]
            y_train, y_test = y[train_index], y[test_index]
            elm = BaseELM(500, C=1e8)
            y_predicted = elm.fit(X_train, y_train).predict(X_test)
            y_pres.append(y_predicted)
            y_tests.append(y_test)
    ca_, oa, aa, kappa = p.save_res_4kfolds_cv(np.asarray(y_pres), np.asarray(y_tests), file_name=None, verbose=True)
    all_oa.append(oa)
    all_aa.append(aa)
    all_kappa.append(kappa)
np.savez('./experimental_results/KSC_sparsity-acc.npz', oa=all_oa, aa=all_aa, kappa=all_kappa)
