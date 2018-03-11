"""
Implementation of sparse subspace clustering based on ELM
--------
Formulation:
        arg_min ||C||_1 s.t. H = HC,
                            diag(C)=0
                            H = f(XW + b)^

--------
Optimize:
        Orthogonal Matching Pursuit(OMP) update coefficient one by one
"""
from __future__ import print_function
import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster.spectral import SpectralClustering
from sklearn.linear_model import OrthogonalMatchingPursuit   # OMP compute coef. column by column
from ExtremeLearningMachine.MLRP.classes.ELM_nonlinear_RP import NRP_ELM
import cvxpy as cvx


class ELM_SSC(BaseEstimator, ClassifierMixin):
    """
    """

    def __init__(self, n_hidden, n_clusters, lambda_coef=1):
        self.n_hidden = n_hidden
        self.n_clusters = n_clusters
        self.lambda_coef =lambda_coef

    def fit_predict_omp(self, X, y=None):
        n_sample = X.shape[0]
        H = NRP_ELM(self.n_hidden, sparse=False).fit(X).predict(X)
        C = np.zeros((n_sample, n_sample))
        # solve sparse self-expressive representation
        for i in range(n_sample):
            y_i = H[i]
            H_i = np.delete(H, i, axis=0)
            # H_T = H_i.transpose()  # M x (N-1)
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=int(n_sample * 0.5), tol=1e20)
            omp.fit(H_i.transpose(), y_i)
            #  Normalize the columns of C: ci = ci / ||ci||_ss.
            coef = omp.coef_ / np.max(np.abs(omp.coef_))
            C[:i, i] = coef[:i]
            C[i+1:, i] = coef[i:]
        # compute affinity matrix
        L = 0.5 * (np.abs(C) + np.abs(C.T))  # affinity graph
        # L = 0.5 * (C + C.T)
        self.affinity_matrix = L
        # spectral clustering
        sc = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed')
        sc.fit(self.affinity_matrix)
        return sc.labels_

    def fit_predict_close(self, X, raw_input_=False):
        """
        using close-form solution
        :param X:
        :param raw_input_:
        :return:
        """
        n_sample = X.shape[0]
        if raw_input_ is True:
            H = X
        else:
            H = NRP_ELM(self.n_hidden, sparse=False).fit(X).predict(X)
        C = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            y_i = H[i]
            H_i = np.delete(H, i, axis=0).transpose()
            term_1 = np.linalg.inv(np.dot(H_i.transpose(), H_i) + self.lambda_coef * np.eye(n_sample - 1))
            w = np.dot(np.dot(term_1, H_i.transpose()), y_i.reshape((y_i.shape[0], 1)))
            w = w.flatten()
            #  Normalize the columns of C: ci = ci / ||ci||_ss.
            coef = w / np.max(np.abs(w))
            C[:i, i] = coef[:i]
            C[i + 1:, i] = coef[i:]
        # compute affinity matrix
        L = 0.5 * (np.abs(C) + np.abs(C.T))  # affinity graph
        self.affinity_matrix = L
        # spectral clustering
        sc = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed')
        sc.fit(self.affinity_matrix)
        return sc.labels_

    def fit_predict_cvx(self, X):
        n_sample = X.shape[0]
        H = X  #NRP_ELM(self.n_hidden, sparse=False).fit(X).predict(X)
        C = np.zeros((n_sample, n_sample))
        # solve sparse self-expressive representation
        for i in range(n_sample):
            y_i = H[i]
            H_i = np.delete(H, i, axis=0)
            # H_T = H_i.transpose()  # M x (N-1)
            # omp = OrthogonalMatchingPursuit(n_nonzero_coefs=500)
            # omp.fit(H_i.transpose(), y_i)
            w = cvx.Variable(n_sample-1)
            objective = cvx.Minimize(0.5 * cvx.sum_squares(H_i.transpose() * w - y_i) + 0.5 * self.lambda_coef * cvx.norm(w, 1))
            prob = cvx.Problem(objective)
            result = prob.solve()
            #  Normalize the columns of C: ci = ci / ||ci||_ss.
            ww = np.asarray(w.value).flatten()
            coef = ww / np.max(np.abs(ww))
            C[:i, i] = coef[:i]
            C[i + 1:, i] = coef[i:]
        # compute affinity matrix
        L = 0.5 * (np.abs(C) + np.abs(C.T))  # affinity graph
        # L = 0.5 * (C + C.T)
        self.affinity_matrix = L
        # spectral clustering
        sc = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed')
        sc.fit(self.affinity_matrix)
        return sc.labels_