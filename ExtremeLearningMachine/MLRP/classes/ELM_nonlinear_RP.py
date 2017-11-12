"""
Implementation of nonlinear random projection using ELM(NRP-ELM)
----------------
X_ = XW
where W is orthogonal
"""
import numpy as np
from scipy.linalg import orth
from scipy.special._ufuncs import expit


class NRP_ELM:
    upper_bound = .5
    lower_bound = -.5

    def __init__(self, n_hidden, sparse=False):
        self.n_hidden = n_hidden
        self.sparse = sparse

    def fit(self, X, W=None):
        if W is None:
            # self.orth_W = np.random.uniform(self.lower_bound, self.upper_bound, size=(X.shape[1] + 1, self.n_hidden))
            if self.sparse is False:
                W = np.random.uniform(self.lower_bound, self.upper_bound, size=(X.shape[1] + 1, self.n_hidden))
            else:
                v1, v2 = (3. / self.n_hidden) ** 0.5, -(3. / self.n_hidden) ** 0.5
                W = np.zeros((X.shape[1] + 1, self.n_hidden))
                W = W.reshape(-1)
                # self.W = np.random.uniform(self.lower_bound, self.upper_bound, size=(X.shape[1] + 1, self.n_hidden))
                W[np.random.choice(range(0, W.size), int(1. / 6. * W.size))] = v1
                W[np.random.choice(np.nonzero(W == 0.)[0], int(1. / 6. * W.size))] = v2
                W = W.reshape(X.shape[1] + 1, self.n_hidden)

            # orthogonalize W and b by QR factorization
            if X.shape[1] >= self.n_hidden:
                self.orth_W = orth(W)
            else:
                self.orth_W = orth(W.transpose()).transpose()
            while self.orth_W.shape[1] < self.n_hidden:
                self.orth_W = np.hstack((self.orth_W, np.zeros((self.orth_W.shape[0], 1))))
        else:
            self.orth_W = W
        return self

    def predict(self, X):
        X_ = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        output = expit(np.dot(X_, self.orth_W))
        return output

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import normalize
# X, y = load_iris(return_X_y=True)
# X = normalize(X)
#
# elm_ae = RP(100, sparse=False)
# elm_ae.fit(X)
# X_transform = elm_ae.predict(X)
# X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.4, random_state=42, stratify=y)
#
# from sklearn.neighbors import KNeighborsClassifier as KNN
# knn_1 = KNN(3)
# knn_1.fit(X_train, y_train)
# y_1 = knn_1.predict(X_test)
#
# # knn_2 = KNN(3)
# # knn_2.fit(X_train, y_train)
# # y_2 = knn_2.predict(X_test)
# acc = accuracy_score(y_test, y_1)
# print(acc)
