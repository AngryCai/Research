"""
A test code for multilayer random projection(MLRP)
"""
import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.linear_model import RidgeClassifier
from ELM import BaseELM
from ExtremeLearningMachine.MLRP.classes.ELM_nonlinear_RP import NRP_ELM


class MLRP_Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_hidden_list):
        self.n_hidden_list = n_hidden_list

    def fit(self, X, y):
        from sklearn.preprocessing import LabelBinarizer
        if np.unique(y).shape[0] == 2:
            Y = self.__one2array(y, 2)
        else:
            Y = LabelBinarizer().fit_transform(y)
        n_layer = self.n_hidden_list.__len__()
        X_ = X # minmax_scale(X)
        ext_X = [[]]*(n_layer + 1)
        ext_X[0] = X_
        spr_model = []
        for i in range(n_layer):
            # nonlinear
            X_rp = np.zeros((X.shape[0], self.n_hidden_list[i]))
            rp_layer = []
            weights = 1./(np.e ** np.arange(1, i + 2))
            for j in range(i + 1):
                rp = NRP_ELM(self.n_hidden_list[i], sparse=True)  # SparseRandomProjection(n_components=self.n_hidden_list[i])
                rp.fit(ext_X[j])
                X_rp_layer = rp.predict(ext_X[j])
                X_rp += weights[j] * X_rp_layer#expit(X_rp_layer)
                rp_layer.append(rp)
            # X_rp = minmax_scale(X_rp)
            ext_X[i+1] = X_rp
            spr_model.append(rp_layer)
        self.spr_model = spr_model
        self.X = ext_X

        # fit a linear classifier
        # clf = RidgeClassifier()  # LinearSVC()
        # clf.fit(ext_X[-1], y)
        # self.clf = clf
        self.B = np.dot(linalg.pinv(self.X[-1]), Y)
        return self

    def predict(self, X):
        n_layer = self.n_hidden_list.__len__()
        X_ = X #minmax_scale(X)
        ext_X = [[]] * (n_layer + 1)
        ext_X[0] = X_
        for i in range(n_layer):
            # nonlinear
            X_rp = np.zeros((X.shape[0], self.n_hidden_list[i]))
            weights = 1. / (np.e ** np.arange(1, i + 2))
            for j in range(i + 1):
                X_rp_layer = self.spr_model[i][j].predict(ext_X[j])
                X_rp += weights[j] * X_rp_layer  # expit(X_rp_layer)
            # X_rp = minmax_scale(X_rp)
            ext_X[i + 1] = X_rp
        # y = self.clf.predict(ext_X[-1])
        output = np.dot(ext_X[-1], self.B)
        y = output.argmax(axis=1)
        return y

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def __one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

"""

## test RP and multi-RP
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = load_iris(return_X_y=True)
norm_X = minmax_scale(X)


# X_train, X_test, y_train, y_test = train_test_split(norm_X, y, test_size=0.4, random_state=42)

n = 50
n_hidden_1 = [n]*50  # range(50, 500, 50)#

clf = [
    LinearSVC(),
    KNeighborsClassifier(),
    RidgeClassifier(),
    BaseELM(n),
    MLRP_Classifier(n_hidden_1)
]
acc = []
for c in clf:
    temp_acc = cross_val_score(c, X, y, cv=3)
    acc.append(np.asarray(temp_acc).mean())

print acc
# mlrp = MLRP_Classifier(n_hidden_1)
# mlrp.fit(X_train, y_train)
# y_pre = mlrp.predict(X_test)
# print accuracy_score(y_test, y_pre)

"""