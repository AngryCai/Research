import numpy as np
import copy

class Layer(object):
    def __init__(self, base_elm, n_unit=100):
        self.n_unit = n_unit
        self.baseline = [copy.deepcopy(base_elm) for i in range(n_unit)]

    def create_layer(self,X_input, y_input):
        self.X_input = X_input
        self.y_input = y_input
        self.train_output = np.zeros((self.X_input.shape[0], self.n_unit*self.y_input.shape[1]))
        temp = np.zeros((self.n_unit, self.X_input.shape[0], self.y_input.shape[1]))
        for i in range(self.n_unit):
            temp[i] = self.baseline[i].fit(X_input, y_input).predict(self.X_input, prob=True)
        for i in range(self.X_input.shape[0]):
            self.train_output[i] = temp[:, i,:].flatten()
        return self

    def predict(self, X):
        self.predict_output = np.zeros((X.shape[0], self.n_unit * self.y_input.shape[1]))
        temp = np.zeros((self.n_unit, X.shape[0], self.y_input.shape[1]))
        for i in range(self.n_unit):
            temp[i] = self.baseline[i].predict(X, prob=True)
        for i in range(X.shape[0]):
            self.predict_output[i] = temp[:, i, :].flatten()
        return self.predict_output



# import sklearn.datasets as dt
# from sklearn.preprocessing import normalize
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler
# from ExtremeLearningMachine.MyELM.ELM import BaseELM
#
# iris = dt.load_iris()
# X, y = iris.get('data'), iris.get('target')  # start with 0
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
#
# lb = preprocessing.LabelBinarizer()
# Y_train = lb.fit_transform(y_train)
# # Y_test = lb.fit_transform(y_test)
#
# layer_1 = Layer(BaseELM(5), n_unit=500).create_layer(X_train, Y_train)
# train_output_1 = layer_1.train_output
#
# layer_2 = Layer(BaseELM(500), n_unit=1).create_layer(train_output_1, Y_train)
# train_output_2 = layer_2.train_output
#
# # layer_3 = Layer(BaseELM(100), n_unit=1).create_layer(train_output_2, Y_train)
# # train_output_3 = layer_3.train_output
#
# predict_output_1 = layer_1.predict(X_test)
# predict_output_2 = layer_2.predict(predict_output_1)
# # predict_output_3 = layer_3.predict(predict_output_2)
#
# y_pre = predict_output_2.argmax(axis=1)
# print 'Accuracy:', accuracy_score(y_test, y_pre)
# print 'predicted labels:', y_pre
# print 'actual labels:', y_pre - y_test
#
#
# from ELM import BaseELM
# e = BaseELM(10)
# e.fit(X_train, y_train)
# y_p = e.predict(X_test)
# print accuracy_score(y_test, y_p)
