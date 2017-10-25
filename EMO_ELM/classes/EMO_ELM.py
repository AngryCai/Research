
from __future__ import print_function
from platypus import NSGAII, DTLZ2, MOEAD, Problem, Real, Binary, nondominated
import numpy as np
from scipy import linalg
from scipy.special._ufuncs import expit
import copy
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold
from Diff_Evo import DiffEvolOptimizer


class EMO_ELM:

    """
    Inspired by Sparse Auto-Encoder, optimize hidden parameters by MOEA
    ---------
    Formulas:
        F(B) = min{ error, KL(p||p') }
    """
    upper_bound = 1.
    lower_bound = -1.

    def __init__(self, n_hidden, sparse_degree=0.05, mu=0.5, n_pop=50, max_iter=1000):
        """

        :param n_hidden:
        :param sparse_degree: desired sparsity
        :param mu: objective balance weight
        :param max_iter:
        """
        self.n_hidden = n_hidden
        self.max_iter = max_iter
        self.sparse_degree = sparse_degree
        self.mu = mu
        self.n_pop = n_pop

    def fit(self, X, y):
        X, y, = copy.deepcopy(X), copy.deepcopy(y)
        self.y = y
        y_bin = self.one2array(y, np.unique(y).shape[0])
        self.classes_ = np.arange(y_bin.shape[1])
        self.n_classes_ = self.classes_.__len__()
        self.X, self.y_bin = X, y_bin
        # # start evolving in MOEA
        num_variables = (self.X.shape[1] + 1) * self.n_hidden
        algorithm = MOEAD(Objectives(num_variables, 2, self.X, y_bin, self.n_hidden, sparse_degree=self.sparse_degree), population_size=self.n_pop, neighborhood_size=5)
        algorithm.run(self.max_iter)
        self.evo_result = algorithm.result
        print('total solution:', algorithm.result.__len__())
        result = nondominated(algorithm.result)
        print('nondominated solution:', result.__len__())
        self.solution = result
        self.W = []
        self.B = []
        self.voting_weight = []
        for s in result:
            W = np.asarray(s.variables).reshape(self.X.shape[1] + 1, self.n_hidden)
            X_ = np.append(self.X, np.ones((self.X.shape[0], 1)), axis=1)
            H = expit(np.dot(X_, W))
            B = np.dot(linalg.pinv(H), y_bin)
            voting_w_ = 1./(s.objectives[0] + 10e-5) * self.mu + 1. / (s.objectives[1] + 10e-5) * (1 - self.mu)
            self.voting_weight.append(voting_w_)
            self.W.append(W)
            self.B.append(B)
        self.voting_weight = np.asarray(self.voting_weight)
        self.W = np.asarray(self.W)
        self.B = np.asarray(self.B)
        return self

    def save_hidden_parameter(self, file_name='moea-hidden-paras'):
        """
        save ELM's hidden parameters W which computed according to non-dominanted solution
        :param file_name:
        :return:
        """
        np.savez(file_name + '.npz', W=self.W, B=self.B)

    def save_evo_result(self, file_name='moea-results'):
        """
        save optimized results that contains non-dominanted solution and its objective values
        :param file_name:
        :return:
        """
        objectives = []
        solutions = []
        for s in self.evo_result:
            objectives.append(s.objectives)
            solutions.append(s.variables)

        objectives = np.asarray(objectives)
        solutions = np.asarray(solutions)
        np.savez(file_name+'.npz', obj=objectives, solution=solutions)

    def get_best_W_index(self):
        """
        get best W according to minimum train error (f2)
        :return:
        """
        s, v = [], []
        for so in self.solution:
            s.append(so.objectives[1])
            v.append(so.variables)
        s = np.asarray(s)
        v = np.asarray(v)
        # W = self.W[s.argmin()]
        return s.argmin()

    def best_predict(self, X):
        index = self.get_best_W_index()
        W = self.W[index]
        B = self.B[index]
        X = copy.deepcopy(X)
        X_test = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        H = expit(np.dot(X_test, W))
        output = np.dot(H, B)
        return output.argmax(axis=1)

    def voting_predict(self, X, y=None):
        X = copy.deepcopy(X)
        X_test = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        # # # -------- compute model weights-------------
        # self.model_weight = []
        # for i in range(self.B.shape[0]):
        #     w = 1. / self.__get_error(np.append(self.X, np.ones((self.X.shape[0], 1)), axis=1), self.y, self.W[i], self.B[i])
        #     self.model_weight.append(w)
        # self.model_weight = np.asarray(self.model_weight)

        # # -------- voting for final results-------------
        y_final, y_pre_ = [], []
        for i in range(self.B.shape[0]):
            y_, acc, kl, avg_activation = self.__get_info(X_test, self.W[i], self.B[i], y)
            y_pre_.append(y_)
            if y is not None:
                print('NO.', i, ': obj values=', self.solution[i].objectives, 'acc=', acc,
                      ' KL=', kl, ' AVG activation:', avg_activation)
        y_pre_ = np.asarray(y_pre_)
        # y_final = y_pre_.mean(axis=0).argmax(axis=1)
        y_final = self.__voting(y_pre_)
        return y_final

    def __voting(self, predicted_ys):
        y = []
        for s in range(predicted_ys.shape[1]):
            temp = np.zeros(self.classes_.shape[0])
            for c in range(self.classes_.shape[0]):
                index = np.nonzero(predicted_ys[:, s] == c)
                temp[c] = self.voting_weight[index].sum()
            y.append(self.classes_[temp.argmax()])
        y = np.asarray(y)
        return y

    def __get_info(self, X_test, W, B, y_test=None):
        H = expit(np.dot(X_test, W))
        y_ = np.dot(H, B).argmax(axis=1)
        # # calculate accuracy
        acc = None
        if y_test is not None:
            acc = accuracy_score(y_test, y_)
        # # calculate KL divergence
        kl = 0.
        desired_degree = np.array([self.sparse_degree, ] * self.n_hidden)
        real_degree = H.mean(axis=0)  # n_hidden dim
        avg_activation = real_degree.mean()
        for p1, p2 in zip(real_degree, desired_degree):
            entr = entropy([p1, 1 - p1], [p2, 1 - p2])
            kl += entr
        return y_, acc, kl, avg_activation

    def get_result(self, key):
        paras = {'W': self.W, 'B': self.B, 'evo_result': self.evo_result}
        return paras[key]

    def __get_error(self, X, y, W, B):
        H = expit(np.dot(X, W))
        y_ = np.dot(H, B).argmax(axis=1)
        error = 1.001 - accuracy_score(y, y_)
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
    def __init__(self, num_variables, num_objectives, X, T, n_hidden, sparse_degree=0.05, num_constraints=0):
        super(Objectives, self).__init__(num_variables, num_objectives, num_constraints) # the number of decision variables, objectives, and constraints
        self.types[:] = Real(-1., 1.)  # specify the type(coding scheme) of decision variables
        self.X, self.T = X, T
        self.n_hidden = n_hidden
        self.sparse_degree = sparse_degree
        self.classes_ = np.arange(T.shape[1])
        self.n_classes_ = self.classes_.__len__()

    def evaluate(self, solution):
        """
        It is the evaluation function that return a value vector of objectives
        :param solution:  also called individual
        :return: None
        """
        X = np.append(self.X, np.ones((self.X.shape[0], 1)), axis=1)
        W = np.asarray(solution.variables).reshape(self.X.shape[1] + 1, self.n_hidden)
        # ## TODO: obj_1: min RSE
        # obj1, obj2 = self.obj_func_MSE(X, W, n_splits=3)
        obj1, obj2 = self.obj_func_cv_error(X, W, n_splits=3)
        solution.objectives[:] = obj1  # self.obj_1(X, W)
        # ## TODO: obj_2: min L0 norm
        solution.objectives[1] = obj2

    def obj_func_cv_error(self, X, W, n_splits=3):
        """
        error
        :param X:
        :param W:
        :param n_splits:
        :return:
        """
        # ## TODO: train learner and calculate error of cross-over validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=233)
        error = 0
        kl = 0.
        desired_degree = np.array([self.sparse_degree, ] * self.n_hidden)
        for train_index, test_index in skf.split(self.X, self.T.argmax(axis=1)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = self.T[train_index], self.T[test_index]
            H = expit(np.dot(X_train, W))
            B = np.dot(linalg.pinv(H), y_train)
            H_ = expit(np.dot(X_test, W))
            output = np.dot(H_, B)
            error += (1 - accuracy_score(y_test.argmax(axis=1), output.argmax(axis=1)))
            kl_split = 0.
            real_degree = H.mean(axis=0)  # n_hidden dim
            for p1, p2 in zip(real_degree, desired_degree):
                entr = entropy([p1, 1 - p1], [p2, 1 - p2])
                kl_split += entr
            kl += kl_split
        return kl/n_splits, error/n_splits

    def obj_func_MSE(self, X, W, n_splits=3):
        """
        error
        :param X:
        :param W:
        :param n_splits:
        :return:
        """
        # ## TODO: train learner and calculate error of cross-over validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=233)
        error = 0
        kl = 0.
        desired_degree = np.array([self.sparse_degree, ] * self.n_hidden)
        for train_index, test_index in skf.split(self.X, self.T.argmax(axis=1)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = self.T[train_index], self.T[test_index]
            H = expit(np.dot(X_train, W))
            B = np.dot(linalg.pinv(H), y_train)
            H_ = expit(np.dot(X_test, W))
            output = np.dot(H_, B)
            # error += (1 - accuracy_score(y_test.argmax(axis=1), output.argmax(axis=1)))
            error += mean_squared_error(y_test, output)
            kl_split = 0.
            real_degree = H.mean(axis=0)  # n_hidden dim
            for p1, p2 in zip(real_degree, desired_degree):
                entr = entropy([p1, 1 - p1], [p2, 1 - p2])
                kl_split += entr
            kl += kl_split
        return kl/n_splits, error/n_splits


class DE_ELM:
    """
    optimize hidden parameters by differential evolution algorithm
    """
    def __init__(self, n_hidden, n_pop=100, max_iter=100):
        self.n_hidden = n_hidden
        self.n_pop = n_pop
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y, = copy.deepcopy(X), copy.deepcopy(y)
        self.y = y
        y_bin = self.one2array(y, np.unique(y).shape[0])
        self.classes_ = np.arange(y_bin.shape[1])
        self.n_classes_ = self.classes_.__len__()
        self.X, self.y_bin = X, y_bin
        bounds = np.array([[-1., 1.], ] * (self.X.shape[1] + 1) * self.n_hidden)
        de = DiffEvolOptimizer(self.__fitness_func, bounds, self.n_pop, F=0.7, C=0.5)
        solution, fitness = de.optimize(ngen=self.max_iter)
        self.W = np.asarray(solution).reshape((self.X.shape[1] + 1, self.n_hidden))
        X_ = np.append(self.X, np.ones((self.X.shape[0], 1)), axis=1)
        H = expit(np.dot(X_, self.W))
        self.B = np.dot(linalg.pinv(H), y_bin)
        return self

    def __fitness_func(self, vars):
        # cost function or optimized function
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=233)
        error = 0
        X_ = np.append(self.X, np.ones((self.X.shape[0], 1)), axis=1)
        W = np.asarray(vars).reshape((self.X.shape[1] + 1, self.n_hidden))
        for train_index, test_index in skf.split(X_, self.y):
            X_train, X_test = X_[train_index], X_[test_index]
            y_train, y_test = self.y_bin[train_index], self.y_bin[test_index]
            H = expit(np.dot(X_train, W))
            B = np.dot(linalg.pinv(H), y_train)
            H_ = expit(np.dot(X_test, W))
            output = np.dot(H_, B)
            error += (1. - accuracy_score(y_test.argmax(axis=1), output.argmax(axis=1)))
            # error += mean_squared_error(y_test, output)
        return error / 3.

    def predict(self, X):
        X = copy.deepcopy(X)
        X_ = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        H = expit(np.dot(X_, self.W))
        output = np.dot(H, self.B)
        return output.argmax(axis=1)

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected




