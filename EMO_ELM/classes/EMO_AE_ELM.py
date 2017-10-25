
from __future__ import print_function
from platypus import NSGAII, DTLZ2, MOEAD, Problem, Real, Binary, nondominated
import numpy as np
from scipy import linalg
from scipy.special._ufuncs import expit
import copy
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale
import scipy.interpolate as interpolate
from Diff_Evo import DiffEvolOptimizer
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.models import Sequential
import theano
from scipy.interpolate import UnivariateSpline

class EMO_AE_ELM:

    """
    Inspired by Sparse Auto-Encoder, optimize hidden parameters by MOEA.
    Expend to ELM Auto-Encoder
    ---------
    Formulas:
        F(B) = min{ RMSE, KL(p||p') }
    """
    upper_bound = 1.
    lower_bound = -1.

    def __init__(self, n_hidden, sparse_degree=0.05, max_iter=1000, n_pop=100, mu=0.5):
        """

        :param n_hidden:
        :param sparse_degree: desired sparsity
        :param mu: objective balance weight
        :param max_iter:
        """
        self.n_hidden = n_hidden
        self.max_iter = max_iter
        self.sparse_degree = sparse_degree
        self.n_pop = n_pop
        self.mu = mu

    def fit(self, X, y):
        X, y, = copy.deepcopy(X), copy.deepcopy(y)
        self.y = y
        y_bin = y
        self.X, self.y_bin = X, y
        # # start evolving in MOEA
        num_variables = (self.X.shape[1] + 1) * self.n_hidden
        algorithm = NSGAII(Objectives(num_variables, 2, self.X, y_bin, self.n_hidden,
                                      sparse_degree=self.sparse_degree), population_size=self.n_pop)

            # MOEAD(Objectives(num_variables, 2, self.X, y_bin, self.n_hidden, sparse_degree=self.sparse_degree),
            #               population_size=self.n_pop, neighborhood_size=int(self.n_pop/10))  # delta=0.5, eta=0.8
        algorithm.run(self.max_iter)
        self.evo_result = algorithm.result
        print('total solution:', algorithm.result.__len__())
        nondom_result = nondominated(algorithm.result)
        print('nondominated solution:', nondom_result.__len__())
        self.nondom_solution = nondom_result
        self.W = []
        self.B = []
        for i in range(nondom_result.__len__()):
            s = nondom_result[i]
            W = np.asarray(s.variables).reshape(self.X.shape[1] + 1, self.n_hidden)
            X_ = np.append(self.X, np.ones((self.X.shape[0], 1)), axis=1)
            H = expit(np.dot(X_, W))
            B = np.dot(linalg.pinv(H), y_bin)
            self.W.append(W)
            self.B.append(B)
            real_degree = H.mean(axis=0)  # n_hidden dim
            avg_activation = real_degree.mean()
            print ('NO.', i, '  obj:', s.objectives, 'AVG activation:', avg_activation)
        self.W = np.asarray(self.W)
        self.B = np.asarray(self.B)
        # # best W/B
        best_index = self.get_best_index()
        self.best_W = self.W[best_index]
        self.best_B = self.B[best_index]
        return self

    def save_hidden_parameter(self, file_name):
        """
        save ELM's hidden parameters W which computed according to non-dominanted solution
        :param file_name:
        :return:
        """
        np.savez(file_name, W=self.W, B=self.B)

    def save_evo_result(self, file_name):
        """
        save optimized results that contains non-dominanted solution and its objective values
        :param file_name:
        :return:
        """
        objectives = []
        solutions = []
        for s in self.nondom_solution:  # self.evo_result
            objectives.append(s.objectives)
            solutions.append(s.variables)
        objectives = np.asarray(objectives)
        solutions = np.asarray(solutions)

        objectives_ = []
        solutions_ = []
        for s in self.evo_result:  # self.evo_result
            objectives_.append(s.objectives)
            solutions_.append(s.variables)
        objectives_ = np.asarray(objectives_)
        solutions_ = np.asarray(solutions_)

        np.savez(file_name, non_obj=objectives, non_solution=solutions,
                 all_obj=objectives_, all_solution=solutions_, W=self.W, B=self.B)

    def get_best_index(self):
        """
        get best W according to minimum train error (f2)
        :return:
        """
        # B_norm = []
        # for B in self.B:
        #     B_norm.append(np.linalg.norm(B))
        # index = np.asarray(B_norm).argmin()
        objectives = []
        solutions = []
        for s in self.nondom_solution:  # self.evo_result
            objectives.append(s.objectives)
            solutions.append(s.variables)
        objectives = np.asarray(objectives)
        solutions = np.asarray(solutions)
        index = self.curvature_splines(objectives[:, 0], objectives[:, 1])
        print('find minimum solution at index ', index)
        return index

    def predict(self, X, W=None):
        X_ = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        if W is None:
            H = expit(np.dot(X_, self.best_W))
        else:
            H = expit(np.dot(X_, W))
        return H

    def predict_linear(self, X):
        """
        transform X use transpose of W: X_ = XB'
        :param X:
        :return:
        """
        return np.dot(X, self.best_B.transpose())

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

    def get_result(self):
        objectives_non = []
        solutions_non = []
        for s in self.nondom_solution:  # self.evo_result
            objectives_non.append(s.objectives)
            solutions_non.append(s.variables)
        objectives_non = np.asarray(objectives_non)
        solutions_non = np.asarray(solutions_non)

        objectives_all = []
        solutions_all = []
        for s in self.evo_result:  # self.evo_result
            objectives_all.append(s.objectives)
            solutions_all.append(s.variables)
        objectives_all = np.asarray(objectives_all)
        solutions_all = np.asarray(solutions_all)

        paras = {'W': self.W, 'B': self.B, 'all_obj': objectives_all,
                 'all_solution': solutions_all, 'non_obj': objectives_non, 'non_solution': solutions_non}
        return paras

    def __get_error(self, X, y, W, B):
        H = expit(np.dot(X, W))
        y_ = np.dot(H, B).argmax(axis=1)
        error = 1.001 - accuracy_score(y, y_)
        return error

    def curvature_splines(self, x, y):
        """
        Finding the knee area by seeking maximal curvature of the curve which is smoothed using B-Spline interpolation.
        :param x:
        :param y:
        :return:
        """
        norm_x, norm_y = minmax_scale(x), minmax_scale(y)
        original_index = norm_x.argsort()
        xx = norm_x[original_index]
        yy = norm_y[original_index]
        # interpolate
        t, c, k = interpolate.splrep(xx, yy, s=0.05, k=3)
        N = 200
        xmin, xmax = norm_x.min(), norm_x.max()
        fx = np.linspace(xmin, xmax, N)
        sp = interpolate.interpolate.BSpline(t, c, k, extrapolate=False)
        fy = sp(fx)
        # calculate curvature
        f_2 = sp.derivative(2)(fx)
        f_1 = sp.derivative(1)(fx)
        curvature = abs(f_2) / np.power(1. + f_1 ** 2, 3./2.)
        max_index = curvature.argmax()
        max_x = fx[max_index]
        # find the point closet to max_xy in the given solutions
        index_ = np.abs(max_x - norm_x).argmin()
        optimal_index = index_  # original_index[index_]
        return optimal_index

class Objectives(Problem):
    """
    Define Constrained/Unconstrained Problem
    """
    def __init__(self, num_variables, num_objectives, X, T, n_hidden, sparse_degree=0.05, num_constraints=0):
        super(Objectives, self).__init__(num_variables, num_objectives, num_constraints) # the number of decision variables, objectives, and constraints
        self.types[:] = Real(-1., 1.)  # specify the type(coding scheme) of decision variables
        self.X, self.T = X, T
        self.n_hidden = n_hidden
        self.sparse_degree = sparse_degree

    def evaluate(self, solution):
        """
        It is the evaluation function that return a value vector of objectives
        :param solution:  also called individual
        :return: None
        """
        X = np.append(self.X, np.ones((self.X.shape[0], 1)), axis=1)
        W = np.asarray(solution.variables).reshape(self.X.shape[1] + 1, self.n_hidden)
        # ## TODO: obj_1: min RSE
        obj1, obj2 = self.obj_func(X, W, n_splits=3)
        solution.objectives[0] = obj1  # self.obj_1(X, W)
        # ## TODO: obj_2: min KL
        solution.objectives[1] = obj2

    def obj_func(self, X, W, n_splits=3):
        """
        error
        :param X:
        :param W:
        :param n_splits:
        :return:
        """
        # ## TODO: train learner and calculate crossover Root Mean Square Error
        # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=233)
        kf = KFold(n_splits=n_splits)
        error = 0
        kl = 0.
        desired_degree = np.array([self.sparse_degree, ] * self.n_hidden)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = self.T[train_index], self.T[test_index]
            H = expit(np.dot(X_train, W))
            B = np.dot(linalg.pinv(H), y_train)
            H_ = expit(np.dot(X_test, W))
            output = np.dot(H_, B)
            error += np.sqrt(mean_squared_error(y_test, output))
            kl_split = 0.
            real_degree = H.mean(axis=0)  # n_hidden dim
            for p1, p2 in zip(real_degree, desired_degree):
                entr = entropy([p1, 1 - p1], [p2, 1 - p2])
                kl_split += entr
            kl += kl_split
        return kl/n_splits, error/n_splits

    def obj_func_v2(self, X, W, n_splits=3):
        """
        error
        :param X:
        :param W:
        :param n_splits:
        :return:
        """
        # ## TODO: train learner and calculate crossover Root Mean Square Error
        # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=233)
        kf = KFold(n_splits=n_splits)
        error = 0
        kl = 0.
        desired_degree = np.array([self.sparse_degree, ] * self.n_hidden)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = self.T[train_index], self.T[test_index]
            H = expit(np.dot(X_train, W))
            B = np.dot(linalg.pinv(H), y_train)
            H_ = expit(np.dot(X_test, W))
            output = np.dot(H_, B)
            error += np.sqrt(mean_squared_error(y_test, output))
            S = np.linalg.norm(W, ord=1)#np.linalg.norm(np.dot(X_test[:, :-1], B.transpose()), ord=2)/np.linalg.norm(np.dot(X_test[:, :-1], B.transpose()), ord=1)
            kl += S
        return kl/n_splits, error/n_splits


# class Autoencoder:
#
#     def __init__(self, n_hidden, max_iter=100):
#         self.n_hidden = n_hidden
#         self.max_iter = max_iter
#
#     def fit(self, X):
#         model = Sequential()
#         model.add(Dense(self.n_hidden, input_shape=(X.shape[1], ), activation='sigmoid', name='encode'))
#         model.add(Dense(X.shape[1], activation='sigmoid'))
#         model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
#         model.fit(X, X, batch_size=16, epochs=self.max_iter,  verbose=0)
#         self.model = model
#
#     def predict(self, X):
#         intermediate_layer_model = Model(inputs=self.model.input,
#                                          outputs=self.model.get_layer('encode').output)
#         activations = intermediate_layer_model.predict(X)
#         return activations


