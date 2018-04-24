from __future__ import print_function
import sys
sys.path.append('/home/caiyaom/python_codes/')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import minmax_scale
from HE_ELM.classes.HE_ELM import HE_ELM
from HE_ELM.classes.ELM import BaseELM
from HE_ELM.classes.HE_ELM_RSM import HE_ELM_RMS
from HE_ELM.classes.Kernel_ELM import KELM
import scipy.io as sio
import numpy as np
from HE_ELM.classes.RidgeClassifier import RidgeClassifier

from HE_ELM.classes.VotingELM import V_ELM
from HE_ELM.classes.BaggingELM import B_ELM
from HE_ELM.classes.AdaboostELM import Ada_ELM
from Toolbox.Preprocessing import Processor
from sklearn.model_selection import GridSearchCV

path = 'F:\Python\UCIDataset-matlab\UCI_25.mat'
# path = '/home/caiyaom/UCI_datasets/UCI_25.mat'
mat = sio.loadmat(path)
del mat['__version__']
del mat['__header__']
del mat['__globals__']
keys = list(mat.keys())
keys.sort()

''' 
----------------------
algorithm comparision
'''
# keys = ['Iris', 'Wine', 'wdbc', 'cotton', 'vowel', 'yeast', 'soybean']
keys = ['yeast', 'satellite', 'shuttle']
if __name__ == '__main__':
    acc_final = {}
    for key in keys:
        print('processing ', key)
        data = mat[key]
        X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
        classes = np.unique(y)
        for c in classes:
            if np.nonzero(y == c)[0].shape[0] < 10:
                X = np.delete(X, np.nonzero(y == c), axis=0)
                y = np.delete(y, np.nonzero(y == c))
        p = Processor()
        y = p.standardize_label(y)
        X = minmax_scale(X)
        print('num classes:', np.unique(y).__len__(), X.shape)

        # # execute classification
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
        # he_elm = HE_ELM([[BaseELM(10, dropout_prob=0.9)]*5, ]*1, KELM(C=1e-5, kernel='rbf'), is_nonmlize=True)
        accs_outer = []
        n_learner = 30
        n_hidden = 20
        # range_ = range(10) if X.shape[0] >= 1000 else range(20)
        for j in range(10) if X.shape[0] >= 2000 else range(20):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
            base_2 = BaseELM(n_hidden, dropout_prob=None)
            v_estimators = [('estimators-' + str(l), BaseELM(n_hidden, dropout_prob=None))
                            for l in range(n_learner)]
            parameters = {'n_hidden': [20, 50, 100, 150, 200]}
            classifier = [HE_ELM([[BaseELM(20, dropout_prob=0.2)] * 20], RidgeClassifier(), is_nonmlize=False),
                          # V_ELM(v_estimators),
                          # B_ELM(base_2, n_learner=n_learner),
                          # Ada_ELM(base_2, n_learner=n_learner),
                          # GridSearchCV(base_2, parameters, cv=3)
                          ]
            acc_clf = []
            for clf in classifier:
                clf.fit(X_train, y_train)
                y_pre = clf.predict(X_test)
                acc__ = accuracy_score(y_test, y_pre)
                acc_clf.append(acc__)
            accs_outer.append(acc_clf)
        # print('%s learners: %s : ' % (n_learner, acc_inner))
        mean, std = np.round(100 * np.asarray(accs_outer).mean(axis=0), 2), \
                    np.round(100*np.asarray(accs_outer).std(axis=0), 2)
        print('%s, mean:%s, std:%s' % (key, mean, std))
        acc_final[key] = [mean, std]
        np.savez('performance.npz', res=acc_final)
        print('-----------------------------------------------\n')


