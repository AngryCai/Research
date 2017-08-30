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

    @staticmethod
    def calculate_Gini_index(X):
        """
        Gini index or Gini coefficient
        :param X:
        :return:
        """
        row, col = X.shape
        X_abs = np.sort(np.abs(X), axis=1)
        gini = np.zeros(row)
        for i in range(row):
            temp = 0.
            norm = np.linalg.norm(X[i], ord=1)
            for k in range(col):
                for j in range(col):
                    temp += abs(X_abs[i][k] - X_abs[i][j])
            gini[i] = temp / (2 * col * norm)
        return gini.mean()

    @staticmethod
    def gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        # All values are treated equally, arrays must be 1d:
        array = array.flatten()
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
            # array = np.abs(array)
        # Values cannot be 0:
        array += 0.0000001
        # Values must be sorted:
        array = np.sort(array)
        # Index per array element:
        index = np.arange(1, array.shape[0] + 1)
        # Number of array elements:
        n = array.shape[0]
        # Gini coefficient:
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))
# X = np.tri(4, 4)
# print X

# print Helper.gini(X)
# print Helper.calculate_Gini_index(X)
