import numpy as np
import copy

from sklearn.preprocessing import StandardScaler


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
        # self.new_X = StandardScaler().fit_transform(self.new_X)
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

    def dropout(self, x, level):
        if level < 0. or level >= 1:
            raise Exception('Dropout level must be in interval [0, 1[.')
        retain_prob = 1. - level
        sample = np.random.binomial(n=1, p=retain_prob, size=x.shape)
        x *= sample
        x /= retain_prob
        return x