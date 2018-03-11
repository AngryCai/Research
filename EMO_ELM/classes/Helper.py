"""
The auxiliary class contain the computation of sparsity and NMSE
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import minmax_scale
import scipy.interpolate as interpolate

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
        return np.sum(np.linalg.norm(X, ord=2, axis=1)) / np.sum(np.linalg.norm(X, ord=1, axis=1)) # L1 of X is based on feature dimension

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

    @staticmethod
    def get_knee_point_(X):
        """
        max-min eval wrote by YMC
        :param X:
        :return:
        """
        if X.shape[0] < 3:
            raise ValueError('X at least three rows.')
        # exclude the min and max point
        a_indx, b_indx = X[:, 0].argmin(), X[:, 0].argmax()
        X_ = np.delete(X, [a_indx, b_indx], axis=0)
        # X_ = X
        a_indx, b_indx = X_[:, 0].argmin(), X_[:, 0].argmax()
        a, b = X_[a_indx], X_[b_indx]
        X__ = np.delete(X_, [a_indx, b_indx], axis=0)
        len_c = np.linalg.norm(a - b)
        coss = []
        for i in range(X__.shape[0]):
            # calculate the angle between min_ and max_ points
            if X__[i].tolist() == a.tolist() or X__[i].tolist() == b.tolist():
                cos_c = -2.
            else:
                len_a = np.linalg.norm(b - X__[i])
                len_b = np.linalg.norm(a - X__[i])
                cos_c = (len_a**2 + len_b**2 - len_c**2) / (2 * len_a * len_b)
            coss.append(cos_c)
        value = X__[np.argmax(coss)].tolist()
        return X.tolist().index(value)

    @staticmethod
    def curvature_splines(x, y, s=0.02, k=3):
        """
        Finding the knee area by seeking maximal curvature of the curve which is smoothed using B-Spline interpolation.
        :param x:
        :param y:
        :return:
        """
        norm_x, norm_y = minmax_scale(x), minmax_scale(y)
        original_index = norm_x.argsort()
        xx = norm_x[original_index]
        yy = norm_y[original_index]
        # interpolate
        t, c, k = interpolate.splrep(xx, yy, s=s, k=k)
        N = 200
        xmin, xmax = norm_x.min(), norm_x.max()
        fx = np.linspace(xmin, xmax, N)
        sp = interpolate.interpolate.BSpline(t, c, k, extrapolate=False)
        fy = sp(fx)
        # calculate curvature
        f_2 = sp.derivative(2)(fx)
        f_1 = sp.derivative(1)(fx)
        curvature = abs(f_2) / np.power(1. + f_1 ** 2, 3./2.)
        max_index = curvature.argmax()
        max_x = fx[max_index]
        # find the point closet to max_xy in the given solutions
        # index_ = np.abs(max_x - norm_x).argmin()
        index_ = np.abs(max_x - norm_x).argsort()
        optimal_index = index_[:3]  # original_index[index_]
        return curvature, (fx, fy), optimal_index



