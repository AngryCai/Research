from scipy import linalg

from ELM import BaseELM
import numpy as np
class DeepELM:
    """
    1th layer of multiple elm will output a class vector as a new feature, it is input 2nd elm layer
    
    """
    def __init__(self, n_model=100, n_hidden=10):
        self.n_model = n_model
        self.n_hidden = n_hidden

    def fit(self, X, y):
        if y.shape.__len__() == 2:
            self.n_classes = y.shape[1]
        else:
            self.n_classes = np.unique(y).__len__()
        self.layer_1 = self.__make_layer(X, y, self.n_model)
        output_1 = np.asarray([m.predict(X, prob=True) for m in self.layer_1])
        ## layer 2
        temp = np.zeros((X.shape[0], self.n_model*self.n_classes))
        for i in range(X.shape[0]):
            temp[i] = output_1[:, i, :].flatten()
        H2 = temp
        self.B2 = np.dot(linalg.pinv(H2), y)
        # self.elm_2 = BaseELM(int(temp.shape[1] * 2)).fit(temp, y)
        return self

    def __make_layer(self, X, y, n_model):
        layer = [None]*n_model
        for i in range(n_model):
            elm = BaseELM(self.n_hidden)
            elm.fit(X, y)
            layer[i] = elm
        return layer

    def predict(self, X):
        output_1 = np.asarray([m.predict(X, prob=True) for m in self.layer_1])
        H2 = np.zeros((X.shape[0],  self.n_classes * self.n_model))
        for i in range(X.shape[0]):
            H2[i] = output_1[:, i, :].flatten()
        output = np.dot(H2, self.B2)
        # output = self.elm_2.predict(H2, prob=True)
        return output.argmax(axis=1)

# """
# ----------
# test
# """
#
# import sklearn.datasets as dt
# from sklearn.preprocessing import normalize
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler
#
#
# iris = dt.load_iris()
# X, y = iris.get('data'), iris.get('target')  # start with 0
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)
#
# lb = preprocessing.LabelBinarizer()
# Y_train = lb.fit_transform(y_train)
# # Y_test = lb.fit_transform(y_test)
#
# eml = ELM_CV(n_model=1000, n_hidden=1)
# eml.fit(X_train, Y_train)
# labels = eml.predict(X_test)
#
# print 'Accuracy:', accuracy_score(y_test, labels)
# print 'predicted labels:', labels
# print 'actual labels:', y_test-labels
# #
# #
# # print 'AdBoost ELM:', accuracy_score(y_test, y_pre_elm_ab)
#
# from ELM import BaseELM
# e = BaseELM(10)
# e.fit(X_train, y_train)
# y_p = e.predict(X_test)
# print accuracy_score(y_test, y_p)
#
#
