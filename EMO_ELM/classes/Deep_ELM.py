"""
Use EMO-AEELM to extract features, then stack a set of hidden layers such that obtain a deep model
X--AEELM_1--AEELM_2--AEELM_3--AEELM_..--T
"""
from __future__ import print_function
from scipy import linalg
from scipy.special._ufuncs import expit


from EMO_AE_ELM import EMO_AE_ELM
import copy
import numpy as np

class DeepELM:

    def __init__(self, list_hidden, list_iter, **list_optional):
        """
        these arguments must have a same order and length.
        :param list_hidden: iterable e.g. [100, 50, 25]
        :param list_iter:
        :param list_optional:
        """
        self.list_hidden = list_hidden
        self.list_iter = list_iter
        self.list_hidden = list_hidden
        self.optional = list_optional

    def fit(self, X, y):
        X, y, = copy.deepcopy(X), copy.deepcopy(y)
        self.y = y
        y_bin = self.one2array(y, np.unique(y).shape[0])
        self.classes_ = np.arange(y_bin.shape[1])
        self.n_classes_ = self.classes_.__len__()
        self.X, self.y_bin = X, y_bin

        # TODO:  Extract features
        iter_X = copy.deepcopy(X)
        self.extracted_X = [copy.deepcopy(X)]
        self.W = []
        for i in range(len(self.list_hidden)):
            if self.optional.has_key('sparse_degree'):
                emo_elm = EMO_AE_ELM(self.list_hidden[i], sparse_degree=self.optional['sparse_degree'][i], max_iter=self.list_iter[i], n_pop=50, mu=0.8)
            else:
                emo_elm = EMO_AE_ELM(self.list_hidden[i], sparse_degree=0.05, max_iter=self.list_iter[i], n_pop=50, mu=0.8)
            emo_elm.fit(iter_X, iter_X)
            iter_X = emo_elm.predict(iter_X)
            W = emo_elm.get_result('best_W')
            self.W.append(W)
            self.extracted_X.append(iter_X)
        # TODO: calculate LS classification layer
        self.B = np.dot(linalg.pinv(self.extracted_X[-1]), y_bin)
        return self

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def predict(self, X):
        XX = copy.deepcopy(X)
        H = None
        for i in range(len(self.W)):
            X_ = np.append(XX, np.ones((XX.shape[0], 1)), axis=1)
            H = expit(np.dot(X_, self.W[i]))
            XX = copy.deepcopy(H)
        output = np.dot(H, self.B)
        return output.argmax(axis=1)

#
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
X, y = load_iris(return_X_y=True)
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
deep_elm = DeepELM([100, 100], [500, 5000], sparse_degree=[0.05, 0.05])
deep_elm.fit(X_train, y_train)
y_pre = deep_elm.predict(X_test)
acc = accuracy_score(y_test, y_pre)
print(acc)

