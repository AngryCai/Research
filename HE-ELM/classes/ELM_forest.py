"""
implementation of Extreme Learning Machine
"""
import warnings

import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier


class BaseELM(BaseEstimator, ClassifierMixin):
    upper_bound = 1.
    lower_bound = -1.
    def __init__(self, n_hidden=50, pro=0.5):
        self.n_hidden = n_hidden #np.random.randint(10, 100)  # 10
        self.pro = pro  #0#np.random.random()
        # self.params = {'n_hidden': self.n_hidden, 'dropout_prob': self.dropout_prob}

    def fit(self, X, y, sample_weight=None):
        # check label has form of 2-dim array
        X, y, = copy.deepcopy(X), copy.deepcopy(y)
        self.sample_weight = None
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            y = self.one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()
        self.W = np.random.uniform(self.lower_bound, self.upper_bound, size=(X.shape[1], self.n_hidden))
        if self.pro is not None:
            self.W = self.dropout(self.W, prob=self.pro)
            # X = self.dropout(X, prob=self.dropout_prob)
        self.b = np.random.uniform(self.lower_bound, self.upper_bound, size=self.n_hidden)
        H = expit(np.dot(X, self.W) + self.b)
        # H = self.dropout(H, prob=0.1)
        if sample_weight is not None:
            self.sample_weight = sample_weight / sample_weight.sum()
            extend_sample_weight = np.diag(self.sample_weight)
            inv_ = linalg.pinv(np.dot(
                np.dot(H.transpose(), extend_sample_weight), H))
            self.B = np.dot(np.dot(np.dot(inv_, H.transpose()), extend_sample_weight), y)
        else:
            self.B = np.dot(linalg.pinv(H), y) ## original ELM
        return self

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def predict(self, X):
        X = copy.deepcopy(X)
        H = expit(np.dot(X, self.W) + self.b)
        output = np.dot(H, self.B)
        return output.argmax(axis=1)

    def predict_proba(self, X):
        X = copy.deepcopy(X)
        H = expit(np.dot(X, self.W) + self.b)
        output = np.dot(H, self.B)
        return output

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            out[key] = value
        return out

    def set_params(self, **parameters):
        for key, value in parameters.items():
            if not parameters:
                # Simple optimisation to gain speed (inspect is slow)
                return self
            setattr(self, key, value)
        return self

    def dropout(self, x, prob=0.2):
        if prob < 0. or prob >= 1:
            raise Exception('Dropout level must be in interval [0, 1]')
        retain_prob = 1. - prob
        sample = np.random.binomial(n=1, p=retain_prob, size=x.shape)
        x *= sample
        # x /= retain_prob
        return x

#
'''
----------
ELM test
----------
'''
# import sklearn.datasets as dt
# from sklearn.preprocessing import normalize
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn import preprocessing
# iris = dt.load_iris()
# X, y = iris.get('data'), iris.get('target')  # start with 0
# X = normalize(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
#
# lb = preprocessing.LabelBinarizer()
# Y_train = lb.fit_transform(y_train)
# # Y_test = lb.fit_transform(y_test)
#
# elm = BaseELM(200, C=1e5)
# elm.fit(X_train, Y_train, sample_weight=None)
# labels_tr = elm.predict(X_train)
# labels_ts = elm.predict(X_test)
#
# print 'Train err:', 1-accuracy_score(y_train, labels_tr)
# print 'Test err:', 1-accuracy_score(y_test, labels_ts)
# print 'predicted labels:', labels
# print 'actual labels:', y_test-labels
#
#
# elm_ab = AdaBoostClassifier(ELM(10), algorithm="SAMME", n_estimators=100)
# elm_ab.fit(X_train, y_train)
# y_pre_elm_ab = elm_ab.predict(X_test)
# print 'AdBoost ELM:', accuracy_score(y_test, y_pre_elm_ab)
