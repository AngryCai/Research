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


class ELM_SSC(BaseEstimator, ClassifierMixin):
    """
    """

    def __init__(self, n_hidden, n_clusters):
        self.n_hidden = n_hidden
        self.n_clusters = n_clusters

    def fit_predict(self, X, y=None):
        n_sample = X.shape[0]
        H = NRP_ELM(self.n_hidden, sparse=True).fit(X).predict(X)
        C = np.nonzero((n_sample, n_sample))
        # solve sparse self-expressive representation
        for i in range(n_sample):
            y_i = H[i]
            H_i = np.delete(H, i, axis=0)
            # H_T = H_i.transpose()  # M x (N-1)
            omp = OrthogonalMatchingPursuit()
            omp.fit(H_i, y_i)
            #  Normalize the columns of C: ci = ci / ||ci||_ss.
            C[i, :] = omp.coef_ / np.linalg.norm(omp.coef_, ord='inf')
        C = C.transpose()
        # compute affinity matrix
        L = np.abs(C) + np.abs(C.T)  # affinity graph
        self.affinity_matrix = L
        # spectral clustering
        sc = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', n_init=50, n_jobs=1)
        sc.fit(self.affinity_matrix)
        return sc.labels_
