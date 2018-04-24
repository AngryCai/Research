import math

import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, minmax_scale, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import random


class RSM_ELM(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learner):
        """
        :param base_learner: learner list, allow multi-layer
        """
        self.base_learner = base_learner

    def fit(self, X, y):
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            y = self.one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()

        output = []
        output_C = []
        self.trained_learner = []
        for n in range(len(self.base_learner)):
            ler = copy.deepcopy(self.base_learner[n])
            ler.fit(X, y)
            out = ler.predict(X, prob=True)
            output.append(out)
            max_ = [out.max(axis=1)]*self.n_classes_
            max_ = np.asarray(max_).transpose()
            output_C.append(out.argmax(axis=1) == y.argmax(axis=1))
            self.trained_learner.append(ler)
        output = np.asarray(output)  # shape:n_lear*n_sam*n_clz
        output_C = np.asarray(output_C)  # shape:n_lear*n_sam*n_clz
        # # logistic
        coef_lr = []
        weight_C = []
        for i in range(self.n_classes_):
            XX_ = output[:, :, i].transpose()  # shape:n_sam*n_ler
            yy_ = output_C[:, :, i].transpose()
            lr = LogisticRegression()
            lr.fit(XX_, yy_)
            coef_lr.append(lr.coef_)
            #  weighted sum
            w_ = np.sum(lr.coef_ * output[:, :, i].transpose(), axis=1)
            weight_C.append(w_)
        self.coef_lr = coef_lr

    def predict(self, X, prob=False):
        output = []
        output_C = []
        for n in range(len(self.trained_learner)):
            ler = self.trained_learner[n]
            out = ler.predict(X, prob=True)
            output.append(out)
            max_ = [out.max(axis=1)]*out.shape[0]
            max_ = np.asarray(max_).transpose()
            output_C.append(out == max_)
        output = np.asarray(output)  # shape:n_lear*n_sam*n_clz
        # # logistic
        weight_output = []
        for i in range(self.n_classes_):
            #  weighted sum
            w_ = np.sum(self.coef_lr[i] * output[:, :, i].transpose(), axis=1)
            weight_output.append(w_)
        weight_output = np.asarray(weight_output).transpose()
        if prob is True:
            return weight_output
        else:
            return weight_output.argmax(axis=1)

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected




import sklearn.datasets as dt
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from ELM import BaseELM
from Kernel_ELM import KELM
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
iris = dt.load_iris()
X, y = iris.get('data'), iris.get('target')  # start with 0
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(y_train)
# Y_test = lb.fit_transform(y_test)

he_elm = RSM_ELM([BaseELM(20, dropout_prob=None)]*5)
he_elm.fit(X_train, y_train)
y_pre = he_elm.predict(X_test)
acc = accuracy_score(y_test, y_pre)
print(acc)
