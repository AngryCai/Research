import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RidgeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, coef_=0.1):
        self.coef_ = coef_

    def fit(self, X, y):
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
        W_ridge = np.dot(np.dot(np.linalg.inv(
            self.coef_ * np.eye(X.shape[1]) + np.dot(X.transpose(), X)), X.transpose()), y)
        self.W_ridge = W_ridge
        return self

    def predict(self, X, prob=False):
        pre_ = np.dot(X, self.W_ridge)
        if prob is True:
            return pre_
        else:
            return pre_.argmax(axis=1)

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def get_params(self, deep=True):
        params = {'coef_': self.coef_}
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self