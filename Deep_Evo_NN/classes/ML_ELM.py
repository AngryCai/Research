"""
Multiple Layer ELM based on AE-ELM
"""
from ELM_AE import ELM_AE
import copy
import numpy as np
from scipy.special import expit

class ML_ELM(object):

    def __init__(self, n_layer, n_hidden_list):
        self.n_layer = n_layer
        self.n_hidden_list = n_hidden_list

    def fit(self, X, y):
        # construct ML ELM
        self.X = copy.deepcopy(X)
        temp_X = copy.deepcopy(X)
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            y = self.__one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()

        self.Betas = []
        for l in range(self.n_layer):
            ae = ELM_AE(self.n_hidden_list[l]).fit(temp_X)
            temp_B = ae.B.transpose()
            self.Betas.append(temp_B)
            temp_X = expit(np.dot(temp_X, temp_B))
        self.output_B = np.dot(np.linalg.pinv(temp_X), y)

    def __one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def predict(self, X):
        temp_X = copy.deepcopy(X)
        for l in range(self.n_layer):
            temp_X = expit(np.dot(temp_X, self.Betas[l]))
        output = np.dot(temp_X, self.output_B)
        return output.argmax(axis=1)


'''
----------
ML-ELM test
----------
'''
# import sklearn.datasets as dt
# from sklearn.preprocessing import normalize
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn import preprocessing
# iris = dt.load_iris()
# X, y = iris.get('data'), iris.get('target')  # start with 0
# X = normalize(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
#
# lb = preprocessing.LabelBinarizer()
# Y_train = lb.fit_transform(y_train)
# # Y_test = lb.fit_transform(y_test)
#
# elm = ML_ELM(3, (10, 20, 50))
# elm.fit(X_train, Y_train)
# labels_ts = elm.predict(X_test)
# print accuracy_score(y_test, labels_ts)