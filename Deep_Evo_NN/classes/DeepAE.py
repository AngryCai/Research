from __future__ import print_function

from sklearn.base import BaseEstimator, ClassifierMixin
from Deep_Evo_NN.classes.EvoAE import EMO_AE_ELM
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from Deep_Evo_NN.classes.Softmax import Softmax
import numpy as np
import copy


class DAE(BaseEstimator, ClassifierMixin):
    """
    -----
    multi-AE + KELM
    """
    def __init__(self, n_hidens, ridge_alpha=1., logic_iter=1000, **kwargs):
        self.alpha = ridge_alpha
        self.n_hiddens = n_hidens
        self.logic_iter = logic_iter
        self.args = kwargs

    def fit(self, X, y):
        models = []
        XX = copy.deepcopy(X)
        X_pro = [XX]
        weights = []
        i = 0
        for __hidden in self.n_hiddens:
            instance_emo_elm = EMO_AE_ELM(__hidden, **self.args)
            XX = instance_emo_elm.fit(XX, XX).predict(XX)
            instance_emo_elm.save_evo_result('F:\Python\Deep_Evo_NN\demo\Experiment\evo-results-' + str(i) + 'layer.npz')
            X_pro.append(XX)
            models.append(instance_emo_elm)
            weights.append(instance_emo_elm.best_W)
            i += 1
        # train last layer
        # ridge = Softmax(batch_size=32, epochs=self.logic_iter, learning_rate=0.01, reg_strength=self.alpha)
        ridge = RidgeClassifier(alpha=self.alpha)
        # ridge = LogisticRegression(C=1e5, max_iter=self.logic_iter)
        ridge.fit(X_pro[-1], y)
        models.append(ridge)
        self.models = models
        self.weights = weights

    def predict(self, X, save_x_hat=None):
        XX = copy.deepcopy(X)
        X_pro = [XX]
        for m in self.models:  # Note: the last model is ridge classifier
            XX = m.predict(XX)
            X_pro.append(XX)
        # y_pre = self.models[-1].predict(X_pro[-1])
        if save_x_hat is not None:
            np.savez(save_x_hat, X_hat=X_pro[1:-1], X=X)
        return X_pro[-1]

    def save_model(self, file_name):
        np.savez(file_name, W=self.weights)
        return self.weights

'''
-------------------
Test on Iris data set
'''

# from sklearn.datasets import load_iris
# from sklearn.preprocessing import minmax_scale
# from sklearn.metrics import accuracy_score
# X, y = load_iris(return_X_y=True)
# X = minmax_scale(X)
# n_hiddens = [10, 10, 10]
# dae = DAE(n_hiddens)
# dae.fit(X, y)
# y_pre = dae.predict(X)
# print('acc:', accuracy_score(y, y_pre))