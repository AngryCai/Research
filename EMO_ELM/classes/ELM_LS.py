"""
Remove the random feature mapping layer, compute Beta directly from input X
---------
Formulas:
    XB = T
"""
from __future__ import print_function
from platypus import NSGAII, DTLZ2, MOEAD, Problem, Real, Binary, nondominated
import numpy as np
from scipy import linalg
from scipy.special._ufuncs import expit
import copy
from sklearn.model_selection import train_test_split, StratifiedKFold

class MOEA_ELM:

    upper_bound = 1.
    lower_bound = -1.

    def __init__(self, C=1., max_iter=1000):
        self.max_iter = max_iter
        self.C = C

    def fit(self, X, y):
        X, y, = copy.deepcopy(X), copy.deepcopy(y)
        self.y = y
        y_bin = self.one2array(y, np.unique(y).shape[0])
        self.classes_ = np.arange(y_bin.shape[1])
        self.n_classes_ = self.classes_.__len__()
        self.X, self.y_bin = X, y_bin
        inv = linalg.pinv(np.dot(X.transpose(), X) + self.C * np.eye(X.shape[1]))
        self.B = np.dot(np.dot(inv, X.transpose()), y_bin)
        # # start evolving in MOEA
        # num_variables = self.n_hidden * self.n_classes_
        # algorithm = MOEAD(Objectives(num_variables, 2, H, y_bin), population_size=100)
        # algorithm.run(self.max_iter)
        # temp = []
        # for s in algorithm.result:
        #     temp.append(s.objectives)
        # np.savez('total_objectives.npz', np.asarray(temp))
        #
        # result = nondominated(algorithm.result)
        # self.B = []
        # for s in result:
        #     self.B.append(np.asarray(s.variables).reshape(self.n_hidden, self.n_classes_))
        # self.B = np.asarray(self.B)
        return self

    def predict(self, X, prob=False):
        # decision making
        X = copy.deepcopy(X)
        output = np.dot(X, self.B)
        if prob:
            return output
        return output.argmax(axis=1)

    def predict_voting(self, X):
        self.model_weight = []
        for i in range(self.B.shape[0]):
            w = 1. / self.cv_error(self.X, self.y, self.B[i])
            self.model_weight.append(w)
        self.model_weight = np.asarray(self.model_weight)
        y_final, y_pre_ = [], []
        for i in range(self.B.shape[0]):
            H = expit(np.dot(X, self.W) + self.b)
            y_ = np.dot(H, self.B[i]).argmax(axis=1)
            y_pre_.append(y_)
        y_pre_ = np.asarray(y_pre_)
        for s in range(X.shape[0]):
            temp = np.zeros(self.n_classes_)
            for c in range(self.n_classes_):
                # num_vote = np.count_nonzero(y_pre_[:, s] == c)
                # temp[c] = num_vote
                index = np.nonzero(y_pre_[:, s] == c)
                temp[c] = self.model_weight[index].sum()
            y_final.append(self.classes_[temp.argmax()])
        return y_final

    def cv_error(self, X, y, B):
        H = expit(np.dot(X, self.W) + self.b)
        y_ = np.dot(H, B).argmax(axis=1)
        error = 1.01 - accuracy_score(y, y_)
        return error

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected


class Objectives(Problem):
    '''
    Define Constrained/Unconstrained Problem
    '''
    def __init__(self, num_variables, num_objectives, H, T, num_constraints=0):
        super(Objectives, self).__init__(num_variables, num_objectives, num_constraints) # the number of decision variables, objectives, and constraints
        self.types[:] = Real(-1.0, 1.0) # specify the type(coding scheme) of decision variables
        self.H, self.T = H, T

    def evaluate(self, solution):
        """
        It is the evaluation function that return a value vector of objectives
        :param solution:  also called individual
        :return: None
        """
        B = np.asarray(solution.variables).reshape(self.H.shape[1], self.T.shape[1])
        # ## TODO: obj_1: min RSE
        solution.objectives[:] = self.obj_1(B)
        # ## TODO: obj_2: min L0 norm
        solution.objectives[1] = self.obj_2(B)

    def obj_1(self, B):
        term_1 = ((np.dot(self.H, B) - self.T)**2).sum()
        return term_1

    def obj_2(self, B):
        term_2 = abs(B).sum()
        return term_2


import sklearn.datasets as dt
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from ELM import BaseELM
iris = dt.load_iris()
X, y = iris.get('data'), iris.get('target')  # start with 0
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(y_train)
# Y_test = lb.fit_transform(y_test)
# # ______________________MOEA ELM_____________________________
ours = MOEA_ELM(C=10e-10, max_iter=10000)
ours.fit(X_train, y_train)
label_ours = ours.predict(X_test)
# label_ours = ours.predict_voting(X_test)

print('Ours:', accuracy_score(y_test, label_ours))

# # ______________________Basic ELM_____________________________
elm = BaseELM(10)
elm.fit(X_train, Y_train, sample_weight=None)
labels_tr = elm.predict(X_train)
labels_ts = elm.predict(X_test)

print('Basic ELM:', accuracy_score(y_test, labels_ts))



