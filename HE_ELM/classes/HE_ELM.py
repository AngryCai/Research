import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, minmax_scale, normalize
from sklearn.model_selection import GridSearchCV


class HE_ELM(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learner, classifier, is_nonmlize=False):
        """
        :param base_learner: learner list, allow multi-layer
        :param classifier:
        """
        self.base_learner = base_learner
        self.classifier = classifier
        self.is_nonmlize = is_nonmlize

    def fit(self, X, y):
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            y = self.one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()
        outputs = []
        layer = []
        for i in range(len(self.base_learner)):
            layer_ = Layer(self.base_learner[i], is_nonmlize=self.is_nonmlize)
            if i == 0:  # # input layer
                out = layer_.train_output(X, y, X_extra=None)
            else:
                out = layer_.train_output(X, y, X_extra=outputs[i-1])  # xxxxxx:outputs[i-1]
            outputs.append(out)
            layer.append(layer_)
        # # output layer: classifier
        # # grid search to find best parameters
        clf_layer = Layer([self.classifier], is_nonmlize=self.is_nonmlize)
        out = clf_layer.train_output(X, y, X_extra=outputs[-1], cv=True)   # xxxxxx:outputs[-1]
        layer.append(clf_layer)
        self.layer = layer

    def predict(self, X, prob=False):
        out = self.layer[0].test_output(X, X_extra=None)
        for i in range(1, len(self.layer)):
            out = self.layer[i].test_output(X, X_extra=out)  # xxxxxx:out
        if prob is True:
            return out
        else:
            return out.argmax(axis=1)

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected


class Layer(object):
    def __init__(self, learner_lst, is_nonmlize=False):
        """
        :param learner_lst:
        """
        self.base_learner = copy.deepcopy(learner_lst)
        self.is_nonmlize = is_nonmlize

    def train_output(self, X, Y, X_extra=None, cv=False):
        n_sample, self.n_clz, = Y.shape
        n_learner = len(self.base_learner)
        output = np.zeros((n_sample, self.n_clz*len(self.base_learner)))
        X_ = np.hstack((X, X_extra)) if X_extra is not None else X
        X_ = normalize(X_, axis=0) if self.is_nonmlize is True else X_
        # X_ = minmax_scale(X_) if self.is_nonmlize is True else X_
        self.trained_learner = []
        for i in range(n_learner):
            ler_ = copy.deepcopy(self.base_learner[i])
            if cv is True:
                # # edit following lines for different classifier
                # parameters = {'C': [1e-4, 1e-3, 1e2, 1e-1, 1, 10, 100, 1000, 10000],
                #               'gamma': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
                parameters = {'coef_': [1e-4, 1e-3, 1e2, 1e-1, 1, 10, 100, 1000, 10000]}
                clf = GridSearchCV(ler_, parameters, cv=3)
                clf.fit(X_, Y.argmax(axis=1))
                # print(clf.best_params_)
                self.trained_learner.append(clf.best_estimator_)
            else:
                ler_.fit(X_, Y)
                output[:, i * self.n_clz:(i + 1) * self.n_clz] = ler_.predict(X_, prob=True)
                self.trained_learner.append(ler_)
        return output

    def test_output(self, X, X_extra=None):
        output = np.zeros((X.shape[0], self.n_clz * len(self.base_learner)))
        X_ = np.hstack((X, X_extra)) if X_extra is not None else X
        X_ = normalize(X_, axis=0) if self.is_nonmlize is True else X_
        # X_ = minmax_scale(X_) if self.is_nonmlize is True else X_
        for i in range(len(self.trained_learner)):
            output[:, i*self.n_clz:(i+1)*self.n_clz] = self.trained_learner[i].predict(X_, prob=True)
        return output
