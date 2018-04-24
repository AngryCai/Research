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

path = 'F:\Python\UCIDataset-matlab\UCI_25.mat'
# path = '/home/caiyaom/UCI_datasets/UCI_25.mat'
mat = sio.loadmat(path)
keys = mat.keys()
keys.remove('__version__')
keys.remove('__header__')
keys.remove('__globals__')
keys.sort()
save_name = 'result.npz'
results = {}


''' 
----------------------
algorithm comparision
'''
keys = ['Iris', 'vowel', 'yeast', 'soybean'] # 'Iris', 'Wine', 'wdbc', 'cotton',


if __name__ == '__main__':
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
        # acc_mean, acc_std = [], []
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)

        acc_outer = []
        for n_rsm in range(1, 21):
            acc_clf = []
            for prob in np.arange(0.1, 1.1, 0.1):
                base_1 = BaseELM(50, dropout_prob=prob)
                clf = HE_ELM_RMS([[base_1] * 20, [base_1] * 20], RidgeClassifier(), n_rsm=n_rsm, is_nonmlize=False)
                clf.fit(X_train, y_train)
                y_pre = clf.predict(X_test)
                acc__ = accuracy_score(y_test, y_pre)
                acc_clf.append(acc__)
            acc_outer.append(acc_clf)
            print('#n_srm=%s, res:%s' % (n_rsm, acc_clf))
        print(acc_outer)

        #     acc_mean.append(mean)
        #     acc_std.append(std)
        # print('Dataset=%s\n mean:%s\n, std:%s' % (key, acc_mean, acc_std))

        print('-----------------------------------------------\n')

