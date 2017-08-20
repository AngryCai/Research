"""
implementation of Kernel-based Extreme Learning Machine(KELM)
Description:
    1) KELM is a special ELM, which regard the feature mapping(random layer) is unknown so
    it need not consider the hidden layer parameters
    2) the output can be written as :
    f(x) = K(X_ts,X_tr) [I/C + K(X_tr, X_tr)]^-1 T = f(x) = K(X_ts,X_tr) Alpha
    
"""
import numpy as np
import copy
# from scipy import linalg
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_kernels
from sklearn.ensemble import AdaBoostClassifier

class KELM(BaseEstimator, ClassifierMixin):
    upper_bound = 1
    lower_bound = -1
    def __init__(self, C=10e3, kernel='linear', **kwds):
        self.C = C
        self.kernel = kernel
        self.kwds = kwds

    def fit(self, X, y):
        # check label has form of 2-dim array
        X, y, = copy.deepcopy(X), copy.deepcopy(y)
        self.X = X
        self.y = y
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            self.y = self.one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()
        # , gamma=self.gamma
        K_tr_matrix = pairwise_kernels(X, X, metric=self.kernel, **self.kwds)
        self.alpha = np.dot(np.linalg.inv(np.eye(X.shape[0])/self.C + K_tr_matrix), self.y)
        return self

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def predict(self, X, prob=False):
        X = copy.deepcopy(X)
        K_ts_matrix = pairwise_kernels(X, self.X, metric=self.kernel, **self.kwds)
        output = np.dot(K_ts_matrix, self.alpha)
        if prob is True:
            return output
        return output.argmax(axis=1)

    def get_params(self, deep=True):
        return {'C':self.C, 'kernel':self.kernel}

    def set_params(self, **parameters):
        return self

# lb = preprocessing.LabelBinarizer()

#
# '''
# ----------
# ELM test
# ----------
# '''
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
# eml = KELM(C=10e-10, kernel='rbf')
# eml.fit(X_train, Y_train)
# labels = eml.predict(X_test)
#
# print 'Accuracy:', accuracy_score(y_test, labels)
# print 'predicted labels:', labels
# print 'actual labels:', y_test-labels

