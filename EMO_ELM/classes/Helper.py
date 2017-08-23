"""
The auxiliary class contain the computation of sparsity and NMSE
"""
import numpy as np


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def calculate_sparsity(X):
        """
        sparsity = L2/L1
        :param X:
        :return:
        """
        return np.linalg.norm(X, ord=2) / np.linalg.norm(X.transpose(), ord=1)  # L1 of X is based on feature dimension

    @staticmethod
    def calculate_NMSE(X, X_hat):
        """
        NMSE = 1/N * sum((X_i - Xhat_i)**2) / mean(X) * mean(Xhat)
        :param X:
        :param X_hat:
        :return:
        """
        N = X.shape[0]
        NMSE = 1 / N * np.sum((X - X_hat)**2) / np.mean(X) * np.mean(X_hat)
        return NMSE