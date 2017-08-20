import numpy as np
import copy
from sklearn.preprocessing import StandardScaler

class ELM_MultiSamples(object):
    def __init__(self, base_elm, n_unit=100):
        self.base_elm = base_elm
        self.n_unit = n_unit
        self.baseline = [copy.deepcopy(base_elm) for i in range(n_unit)]

    def fit(self,X_input, y_input):
        self.X_input = X_input
        self.y_input = y_input
        self.n_classes = self.y_input.shape[1]
        n_samples = self.X_input.shape[0]
        self.new_X = np.zeros((n_samples * self.n_unit, self.X_input.shape[1] + self.n_classes))
        self.new_y = np.zeros((n_samples * self.n_unit, self.n_classes))
        for i in range(self.n_unit):
            prob = self.baseline[i].fit(X_input, y_input).predict(self.X_input, prob=True)
            self.new_X[n_samples*i:n_samples * (i + 1)] = np.append(self.X_input, prob, axis=1)
            self.new_y[n_samples*i:n_samples * (i + 1)] = self.y_input

        # train new model
        # self.new_X = StandardScaler().fit_transform(self.new_X)
        self.new_decision_elm= KELM(C=1000)
        self.new_decision_elm.fit(self.new_X, self.new_y)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        new_y = np.zeros((self.n_unit, n_samples, self.n_classes))
        for i in range(self.n_unit):
            prob = self.baseline[i].predict(X, prob=True)
            new_y[i] = prob
            # new_y[:n_samples * (i + 1)] = self.y_input
        new_y = new_y.mean(axis=0)
        new_X = np.append(X, new_y, axis=1)
        new_X = StandardScaler().fit_transform(new_X)
        out = self.new_decision_elm.predict(new_X)
        return out

class MultiELM:
    def __init__(self, base_elm, decision_elm,  n_unit=100):
        self.base_elm = base_elm
        self.n_unit = n_unit
        self.decision_elm = copy.deepcopy(decision_elm)
        self.baseline = [copy.deepcopy(base_elm) for i in range(n_unit)]

    def fit(self,X_input, y_input):
        self.X_input = X_input
        self.y_input = y_input
        self.new_X = copy.deepcopy(X_input)
        for i in range(self.n_unit):
            prob = self.baseline[i].fit(X_input, y_input).predict(self.X_input, prob=True)
            self.new_X = np.append(self.new_X, prob, axis=1)
        # train new model
        self.decision_elm.fit(self.new_X, y_input)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        new_X = copy.deepcopy(X)
        # new_y = np.zeros((self.n_unit, n_samples, self.n_classes))
        for i in range(self.n_unit):
            prob = self.baseline[i].predict(X, prob=True)
            new_X = np.append(new_X, prob, axis=1)
            # new_y[:n_samples * (i + 1)] = self.y_input
        # new_X = StandardScaler().fit_transform(new_X)
        out = self.decision_elm.predict(new_X)
        return out


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
            self.train_output[i] = temp[:, i, :].flatten()
        return self

    def predict(self, X):
        self.predict_output = np.zeros((X.shape[0], self.n_unit * self.y_input.shape[1]))
        temp = np.zeros((self.n_unit, X.shape[0], self.y_input.shape[1]))
        for i in range(self.n_unit):
            temp[i] = self.baseline[i].predict(X, prob=True)
        for i in range(X.shape[0]):
            self.predict_output[i] = temp[:, i, :].flatten()
        return self.predict_output

import sklearn.datasets as dt
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from ExtremeLearningMachine.HE_ELM.ELM import BaseELM
from Kernel_ELM import KELM

iris = dt.load_iris()
X, y = iris.get('data'), iris.get('target')  # start with 0
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(y_train)
# Y_test = lb.fit_transform(y_test)

base_hidden = 5
n_unit = 5
C = 10e3

elm_is = ELM_MultiSamples(BaseELM(base_hidden), n_unit=n_unit).fit(X_train, Y_train)
y_pre = elm_is.predict(X_test)
print 'sample:', accuracy_score(y_test, y_pre)

from ELM import BaseELM
e = BaseELM(base_hidden)
e.fit(X_train, Y_train)
y_p = e.predict(X_test)
print 'Base:', accuracy_score(y_test, y_p)

kelm = KELM(C=C)
kelm.fit(X_train, Y_train)
y_p_ = kelm.predict(X_test)
print 'KELM:', accuracy_score(y_test, y_p_)

melm = MultiELM(BaseELM(base_hidden), KELM(C=C), n_unit=n_unit)
melm.fit(X_train, Y_train)
y_pre_melm = melm.predict(X_test)
print 'Base+KELM:', accuracy_score(y_test, y_pre_melm)


layer_1 = Layer(BaseELM(base_hidden), n_unit=n_unit).create_layer(X_train, Y_train)
train_output_1 = layer_1.train_output
layer_2 = Layer(KELM(C=C, kernel='rbf'), n_unit=1).create_layer(train_output_1, Y_train)  # KELM(C=1000, kernel='rbf')
train_output_2 = layer_2.train_output
predict_output_1 = layer_1.predict(X_test)
predict_output_2 = layer_2.predict(predict_output_1)
y_pre = predict_output_2.argmax(axis=1)
print 'Layer:', accuracy_score(y_test, y_pre)
