import math

import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, minmax_scale, normalize
from sklearn.model_selection import GridSearchCV
import random


class HE_ELM_RMS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learner, classifier, n_rsm=10, is_nonmlize=False):
        """
        :param base_learner: learner list, allow multi-layer
        :param classifier:
        """
        self.base_learner = base_learner
        self.classifier = classifier
        self.n_rsm = n_rsm
        self.is_nonmlize = is_nonmlize

    def fit(self, X, y):
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            y = self.one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()
        outputs = []
        layer = []
        for i in range(len([self.base_learner[0]])):
            layer_ = Concate_Layer(self.base_learner[i], is_nonmlize=self.is_nonmlize)
            if i == 0:  # # input layer
                out = layer_.train_output(X, y, X_extra=None)
            else:
                out = layer_.train_output(X, y, X_extra=outputs[i-1])  # xxxxxx:outputs[i-1]
            outputs.append(out)
            layer.append(layer_)

        # # RSM layer
        rsm_layer = RMS_Layer(self.base_learner[1], n_rsm=self.n_rsm)
        rsm_out = rsm_layer.train_output(X, y, X_extra=outputs[-1])
        outputs.append(rsm_out)
        layer.append(rsm_layer)

        # # output layer: classifier
        # # grid search to find best parameters
        clf_layer = Concate_Layer([self.classifier], is_nonmlize=self.is_nonmlize)
        out = clf_layer.train_output(X, y, X_extra=outputs[-1], cv=True)   # xxxxxx:outputs[-1]
        layer.append(clf_layer)
        self.layer = layer

    def predict(self, X, prob=False):
        out = self.layer[0].test_output(X, X_extra=None)
        for i in range(1, len(self.layer)):
            out = self.layer[i].test_output(X, X_extra=out)  # xxxxxx:out
        if prob is True:
            return out
        else:
            return out.argmax(axis=1)

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected


class Concate_Layer(object):
    def __init__(self, learner_lst, is_nonmlize=False):
        """
        :param learner_lst:
        """
        self.base_learner = copy.deepcopy(learner_lst)
        self.is_nonmlize = is_nonmlize

    def train_output(self, X, Y, X_extra=None, cv=False):
        n_sample, self.n_clz, = Y.shape
        n_learner = len(self.base_learner)
        output = np.zeros((n_sample, self.n_clz*len(self.base_learner)))
        X_ = np.hstack((X, X_extra)) if X_extra is not None else X
        X_ = normalize(X_, axis=0) if self.is_nonmlize is True else X_
        # X_ = minmax_scale(X_) if self.is_nonmlize is True else X_
        self.trained_learner = []
        for i in range(n_learner):
            ler_ = copy.deepcopy(self.base_learner[i])
            if cv is True:
                # # edit following lines for different classifier
                # parameters = {'C': [1e-4, 1e-3, 1e2, 1e-1, 1, 10, 100, 1000, 10000],
                #               'gamma': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
                parameters = {'coef_':[1e-4, 1e-3, 1e2, 1e-1, 1, 10, 100, 1000, 10000]}
                clf = GridSearchCV(ler_, parameters, cv=3)
                clf.fit(X_, Y.argmax(axis=1))
                # print(clf.best_params_)
                self.trained_learner.append(clf.best_estimator_)
            else:
                ler_.fit(X_, Y)
                output[:, i * self.n_clz:(i + 1) * self.n_clz] = ler_.predict(X_, prob=True)
                self.trained_learner.append(ler_)
        return output

    def test_output(self, X, X_extra=None):
        output = np.zeros((X.shape[0], self.n_clz * len(self.base_learner)))
        X_ = np.hstack((X, X_extra)) if X_extra is not None else X
        X_ = normalize(X_, axis=0) if self.is_nonmlize is True else X_
        # X_ = minmax_scale(X_) if self.is_nonmlize is True else X_
        for i in range(len(self.trained_learner)):
            output[:, i*self.n_clz:(i+1)*self.n_clz] = self.trained_learner[i].predict(X_, prob=True)
        return output


class RMS_Layer(object):
    def __init__(self, learner_lst, n_rsm=20, n_fea=None, is_nonmlize=False):
        """

        :param learner_lst:
        :param n_rsm: how many times RSM are conduced
        :param n_fea: how many features will be selected for each RSM
        :param is_nonmlize:
        """
        self.base_learner = copy.deepcopy(learner_lst)
        self.is_nonmlize = is_nonmlize
        self.n_rsm = n_rsm
        self.n_fea = n_fea

    def train_output(self, X, Y, X_extra=None):
        n_sample, self.n_clz, = Y.shape
        n_learner = len(self.base_learner)
        X_ = np.hstack((X, X_extra)) if X_extra is not None else X
        X_ = normalize(X_, axis=0) if self.is_nonmlize is True else X_
        self.n_fea = int(X_.shape[1] * 0.8) if self.n_fea is None else self.n_fea
        max_n_rsm = math.factorial(X_.shape[1]) / math.factorial(self.n_fea) / math.factorial(X_.shape[1] - self.n_fea)
        if self.n_rsm > max_n_rsm:
            self.n_rsm = max_n_rsm
        # X_ = minmax_scale(X_) if self.is_nonmlize is True else X_
        self.trained_learner = []
        indx_lst = [random.sample(range(X_.shape[1]), self.n_fea) for l in range(self.n_rsm)]  # subspace sampling
        self.indx_lst = indx_lst
        X_rsm = X_[:, indx_lst]  # axis: n_sam, n_rsm, n_fea
        super_X = X_rsm[:, 0, :]
        super_Y = Y
        for s in range(1, self.n_rsm):
            super_X = np.vstack((super_X, X_rsm[:, s, :]))  # # shape: increase number of samples
            super_Y = np.vstack((super_Y, Y))
        final_out = np.zeros((n_sample, self.n_clz*len(self.base_learner)*self.n_rsm))
        for i in range(n_learner):
            ler_ = copy.deepcopy(self.base_learner[i])
            # rsm_out = np.zeros((n_sample, self.n_rms * self.n_clz))
            ler_.fit(super_X, super_Y)
            rsm_out = ler_.predict(super_X, prob=True)
            final_out__ = rsm_out[:n_sample, :]
            for j in range(1, self.n_rsm):
                final_out__ = np.hstack((final_out__, rsm_out[n_sample*j:n_sample*(j+1), :]))
            final_out[:,  i * self.n_clz * self.n_rsm:(i + 1) * self.n_clz * self.n_rsm] = final_out__
            self.trained_learner.append(ler_)
        return final_out

    def test_output(self, X, X_extra=None):
        n_sample = X.shape[0]
        X_ = np.hstack((X, X_extra)) if X_extra is not None else X
        X_ = normalize(X_, axis=0) if self.is_nonmlize is True else X_
        X_rsm = X_[:, self.indx_lst]  # axis: n_sam, n_rsm, n_fea
        super_X = X_rsm[:, 0, :]
        final_out = np.zeros((n_sample, self.n_clz * len(self.base_learner) * self.n_rsm))
        for s in range(1, self.n_rsm):
            super_X = np.vstack((super_X, X_rsm[:, s, :]))  # # shape: increase number of samples
        for i in range(len(self.trained_learner)):
            rsm_out = self.trained_learner[i].predict(super_X, prob=True)
            final_out__ = rsm_out[:n_sample, :]
            for j in range(1, self.n_rsm):
                final_out__ = np.hstack((final_out__, rsm_out[n_sample * j:n_sample * (j + 1), :]))
            final_out[:, i * self.n_clz * self.n_rsm:(i + 1) * self.n_clz * self.n_rsm] = final_out__
        return final_out


"""
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

he_elm = HE_ELM_RMS([[BaseELM(20, dropout_prob=0.5)]*5, [BaseELM(20, dropout_prob=0.5)]*5], KELM(C=1, kernel='rbf'))
he_elm.fit(X_train, y_train)
y_pre = he_elm.predict(X_test)
acc = accuracy_score(y_test, y_pre)
print(acc)
"""