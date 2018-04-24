import math

import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, minmax_scale, normalize
from sklearn.model_selection import GridSearchCV
import random
from ELM import BaseELM


class H_ELM_E(BaseEstimator, ClassifierMixin):
    def __init__(self, n_base_learner, n_ensemble, n_hidden=20, is_nonmlize=False):
        """
        :param base_learner: learner list, allow multi-layer
        :param classifier:
        """
        self.n_base_learner = n_base_learner
        self.is_nonmlize = is_nonmlize
        self.n_ensemble = n_ensemble
        self.n_hidden = n_hidden
        self.base_learner = [('estimators-' + str(l), BaseELM(self.n_hidden, dropout_prob=None))
                             for l in range(self.n_base_learner)]

    def fit(self, X, y):
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            y = self.one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()

        v_elm = V_ELM(self.base_learner, voting='soft')

        v_elm_ = copy.deepcopy(v_elm)
        self.layer_1 = Concate_Layer([v_elm_]*self.n_ensemble, is_nonmlize=self.is_nonmlize)
        outputs = self.layer_1.train_output(X, y, X_extra=None)

        self.classifier = copy.deepcopy(v_elm)
        self.classifier.fit(outputs, y)
        # # output layer: classifier
        # # grid search to find best parameters

    def predict(self, X, prob=False):
        out_1 = self.layer_1.test_output(X)
        output = self.classifier.predict_proba(out_1)
        if prob is True:
            return output
        else:
            return output.argmax(axis=1)

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
                ler_.fit(X_, Y.argmax(axis=1))
                output[:, i * self.n_clz:(i + 1) * self.n_clz] = ler_.predict_proba(X_)
                self.trained_learner.append(ler_)
        return output

    def test_output(self, X, X_extra=None):
        output = np.zeros((X.shape[0], self.n_clz * len(self.base_learner)))
        X_ = np.hstack((X, X_extra)) if X_extra is not None else X
        X_ = normalize(X_, axis=0) if self.is_nonmlize is True else X_
        # X_ = minmax_scale(X_) if self.is_nonmlize is True else X_
        for i in range(len(self.trained_learner)):
            output[:, i*self.n_clz:(i+1)*self.n_clz] = self.trained_learner[i].predict_proba(X_)
        return output


class V_ELM:
    def __init__(self, base_learner, voting='hard'):
        self.base_learner = base_learner
        self.voting = voting

    def fit(self, X, y):
        if len(y.shape) != 1:
            y = y.argmax(axis=1)
        eclf = VotingClassifier(estimators=self.base_learner, voting=self.voting)
        eclf.fit(X, y)
        self.eclf = eclf
        return self

    def predict(self, X):
        return self.eclf.predict(X)

    def predict_proba(self, X):
        return self.eclf.predict_proba(X)


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
v_estimators = [('estimators-' + str(l), BaseELM(50, dropout_prob=None))
                                for l in range(5)]

lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(y_train)
# Y_test = lb.fit_transform(y_test)

he_elm = H_ELM_E(10, 5, n_hidden=10)
he_elm.fit(X_train, y_train)
y_pre = he_elm.predict(X_test)
acc = accuracy_score(y_test, y_pre)
print(acc)
"""
