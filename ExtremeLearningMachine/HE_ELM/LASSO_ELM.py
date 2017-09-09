"""
Loss = ||g(XW + b) - T||_2 + C*||W||_21
"""
import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import MultiTaskLasso

class LELM:
    upper_bound = 1.
    lower_bound = -1.

    def __init__(self, n_hidden, C=1., max_iter = 10000):
        self.n_hidden = n_hidden
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        # check label has form of 2-dim array
        X, y, = copy.deepcopy(X), copy.deepcopy(y)
        self.sample_weight = None
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            y = self.__one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()
        self.W = np.random.uniform(self.lower_bound, self.upper_bound, size=(X.shape[1], self.n_hidden))
        self.b = np.random.uniform(self.lower_bound, self.upper_bound, size=self.n_hidden)
        H = expit(np.dot(X, self.W) + self.b)
        self.multi_lasso = MultiTaskLasso(self.C, max_iter=self.max_iter).fit(H, y)

    def __one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def predict(self, X):
        H = expit(np.dot(X, self.W) + self.b)
        output = self.multi_lasso.predict(H)
        return output.argmax(axis=1)


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
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)
#
# lb = preprocessing.LabelBinarizer()
# Y_train = lb.fit_transform(y_train)
# # Y_test = lb.fit_transform(y_test)
#
# elm = LELM(10, C=1e-10, max_iter=10000)
# elm.fit(X_train, Y_train)
# labels_tr = elm.predict(X_train)
# labels_ts = elm.predict(X_test)
#
# print 'Train err:', accuracy_score(y_train, labels_tr)
# print 'Test err:', accuracy_score(y_test, labels_ts)
# print labels_ts
#
#
