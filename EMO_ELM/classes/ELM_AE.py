"""
Autoencoder ELM that used to do feature extraction or dimension reduction
formula:
    ==> X_ = XB'
Note that it is a linear transformation based on output weights. Where W and b are orthogonal matrices.
"""
import numpy as np
from scipy.linalg import orth
from scipy.special import expit

class ELM_AE(object):
    upper_bound = 1.
    lower_bound = -1.
    def __init__(self, n_hidden, activation='sigmoid', sparse=False):
        """

        :param n_hidden:
        :param activation: linear or nonlinear activation function
        """
        self.n_hidden = n_hidden
        self.activation = activation
        self.sparse = sparse

    def fit(self, X):
        if self.sparse is False:
            W = np.random.uniform(self.lower_bound, self.upper_bound, size=(X.shape[1] + 1, self.n_hidden))
        else:
            v1, v2 = (3. / self.n_hidden) ** 0.5, -(3. / self.n_hidden) ** 0.5
            W = np.zeros((X.shape[1] + 1, self.n_hidden))
            W = W.reshape(-1)
            # self.W = np.random.uniform(self.lower_bound, self.upper_bound, size=(X.shape[1] + 1, self.n_hidden))
            W[np.random.choice(range(0, W.size), int(1./6. * W.size))] = v1
            W[np.random.choice(np.nonzero(W == 0.)[0], int(1. / 6. * W.size))] = v2
            W = W.reshape(X.shape[1] + 1, self.n_hidden)

        # orthogonalize W and b by QR factorization
        if X.shape[1] >= self.n_hidden:
            self.orth_W = orth(W)
        else:
            self.orth_W = orth(W.transpose()).transpose()
        X_ = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        if self.activation == 'sigmoid':
            H = expit(np.dot(X_, self.orth_W))
        if self.activation == 'linear':
            H = np.dot(X_, self.orth_W)
        self.B = np.dot(np.linalg.pinv(H), X)
        return self

    def predict(self, X):
        """
        transform initial X use matrix B
        :param X:
        :return:
        """
        # X_ = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        output = np.dot(X, self.B.transpose())
        return output

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import normalize
# X, y = load_iris(return_X_y=True)
# X = normalize(X)
#
# elm_ae = ELM_AE(20, activation='sigmoid', sparse=True)
# elm_ae.fit(X)
# X_transform = elm_ae.predict(X)
# X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.4, random_state=42, stratify=y)
#
# from sklearn.neighbors import KNeighborsClassifier as KNN
# knn_1 = KNN(3)
# knn_1.fit(X_train, y_train)
# y_1 = knn_1.predict(X_test)
# # knn_2 = KNN(3)
# # knn_2.fit(X_train, y_train)
# # y_2 = knn_2.predict(X_test)
# acc = accuracy_score(y_test, y_1)
# print(acc)

