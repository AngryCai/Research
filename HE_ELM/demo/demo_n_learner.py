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
from HE_ELM.classes.H_ELM_E import H_ELM_E

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
keys = ['cotton', 'Iris', 'Wine', ]#, 'car', 'vowel', 'glass'] #

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
        accs_outer = []
        acc_mean, acc_std = [], []
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
        for n_learner in range(5, 101, 10):
            acc_inner = []
            for j in range(10):
                base_2 = BaseELM(20, dropout_prob=None)
                v_estimators = [('estimators-' + str(l), BaseELM(20, dropout_prob=None))
                                for l in range(n_learner)]
                classifier = [HE_ELM_RMS([[BaseELM(20, dropout_prob=0.2)] * n_learner, [BaseELM(20, dropout_prob=0.2)]
                                          * n_learner], RidgeClassifier(), is_nonmlize=False),
                              V_ELM(v_estimators),
                              B_ELM(BaseELM(20, dropout_prob=None), n_learner=n_learner),
                              Ada_ELM(BaseELM(20, dropout_prob=None), n_learner=n_learner),
                              H_ELM_E(10, n_learner, n_hidden=20)
                              ]
                # classifier = [H_ELM_E(10, n_learner, n_hidden=20)]
                acc_clf = []
                for clf in classifier:
                    clf.fit(X_train, y_train)
                    y_pre = clf.predict(X_test)
                    acc__ = accuracy_score(y_test, y_pre)
                    acc_clf.append(acc__)
                acc_inner.append(acc_clf)
            accs_outer.append(acc_inner)
            # print('%s learners: %s : ' % (n_learner, acc_inner))
            mean, std = np.asarray(acc_inner).mean(axis=0), np.asarray(acc_inner).std(axis=0)
            print('#learner=%s, mean:%s, std:%s' % (n_learner, mean, std))
            acc_mean.append(mean)
            acc_std.append(std)
        print('Dataset=%s\n mean:%s\n, std:%s' % (key, acc_mean, acc_std))

        print('-----------------------------------------------\n')

