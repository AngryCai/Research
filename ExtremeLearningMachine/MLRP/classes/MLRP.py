"""
A test code for multilayer random projection(MLRP)
"""
import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import minmax_scale
from sklearn.tree import DecisionTreeClassifier
from ExtremeLearningMachine.MLRP.classes.ELM_nonlinear_RP import NRP_ELM

class MLRP:
    upper_bound = 1.
    lower_bound = -1.

    def __init__(self, n_hidden_list):
        self.n_hidden_list = n_hidden_list

    def fit(self, X, y=None):
        n_layer = self.n_hidden_list.__len__()
        X_ = minmax_scale(X)
        ext_X = [[]]*(n_layer + 1)
        ext_X[0] = X_
        for i in range(n_layer):
            # X_rp = SparseRandomProjection(n_components=self.n_hidden_list[i]).fit_transform(ext_X[i])
            # nonlinear
            X_rp = np.zeros((X.shape[0], self.n_hidden_list[i]))
            for j in range(i + 1):
                # X_rp_layer = SparseRandomProjection(n_components=self.n_hidden_list[i]).fit_transform(ext_X[j])
                # X_rp += expit(X_rp_layer)
                X_rp_layer = NRP_ELM(self.n_hidden_list[i], sparse=False).fit(ext_X[j]).predict(ext_X[j])
                X_rp += X_rp_layer
            X_rp = minmax_scale(X_rp)
            ext_X[i+1] = X_rp
        self.X = ext_X
        return self

    def predict(self, X):
        return self.X

