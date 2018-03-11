"""
Description: implementation of Stacked Auto-encoder(SAE)
Author:  Yaoming Cai
"""
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, regularizers
from keras.models import Sequential
import numpy as np
import copy
from keras.optimizers import SGD
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from Deep_Evo_NN.classes.Softmax import Softmax

class SAE(object):
    def __init__(self, layer_sizes, learn_rate=0.1, epoch=1000, constraint=None):
        self.layer_sizes = layer_sizes
        self.learn_rate = learn_rate
        self.epoch = epoch
        self.constraint = constraint

    def fit(self, X, y):
        models = []
        XX = copy.deepcopy(X)
        X_pro = [XX]
        num_classes = np.unique(y).__len__()
        y_train = keras.utils.to_categorical(y, num_classes)
        weights = []
        for __hidden in self.layer_sizes:
            instance_ae = AE(__hidden, learn_rate=self.learn_rate, max_iter=self.epoch, constraint=self.constraint)
            XX = instance_ae.fit(XX).predict(XX)
            X_pro.append(XX)
            models.append(instance_ae)
            # get weights
            w, b = instance_ae.model.get_weights()[0], instance_ae.model.get_weights()[1]
            W = np.vstack((w, b.reshape((1, b.shape[0]))))
            weights.append(W)

        # train last layer
        logistc_classifier = LogisticRegression(C=1e5, max_iter=self.epoch)
        # logistc_classifier = RidgeClassifier(alpha=1e-6)
        # logistc_classifier = Softmax(batch_size=32, epochs=self.epoch, learning_rate=5, reg_strength=1e5)
        logistc_classifier.fit(X_pro[-1], y)
        models.append(logistc_classifier)
        self.models = models
        self.weights = weights

    def predict(self, X, save_x_hat=None):
        XX = copy.deepcopy(X)
        X_pro = [XX]
        for m in self.models:  # Note: the last model is ridge classifier
            XX = m.predict(XX)
            X_pro.append(XX)
        label = X_pro[-1]  # X_pro[-1].argmax(axis=1)  # self.models[-1].predict_classes(X_pro[-1], verbose=0)
        if save_x_hat is not None:
            np.savez(save_x_hat, X_hat=X_pro[1:-1], X=X)
        return label

    def save_model(self, file_name):
        np.savez(file_name, W=self.weights)
        return self.weights


class AE:
    def __init__(self, n_hidden, learn_rate=0.1, max_iter=1000, constraint=None):
        self.n_hidden = n_hidden
        self.learn_rate = learn_rate
        self.max_iter = max_iter
        self.constraint = constraint

    def fit(self, X):
        model = Sequential()
        if self.constraint == 'L1':
            model.add(Dense(self.n_hidden, input_shape=(X.shape[1], ), activation='sigmoid', name='encode',
                            activity_regularizer=regularizers.l1(1e-6)))
        elif self.constraint == 'L2':
            model.add(Dense(self.n_hidden, input_shape=(X.shape[1],), activation='sigmoid', name='encode',
                            activity_regularizer=regularizers.l2(1e-6)))
        else:
            model.add(Dense(self.n_hidden, input_shape=(X.shape[1],), activation='sigmoid', name='encode'))
        model.add(Dense(X.shape[1], activation='sigmoid'))
        sgd = SGD(lr=self.learn_rate)
        model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
        model.fit(X, X, batch_size=32, epochs=self.max_iter,  verbose=0)
        self.model = model
        return self

    def predict(self, X):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer('encode').output)
        activations = intermediate_layer_model.predict(X)
        return activations


'''
Test 
-----------------------
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
# elm = SAE([50, 50], 1000)
# elm.fit(X_train, y_train)
# labels_ts = elm.predict(X_test)
#
# print 'Test acc:', accuracy_score(y_test, labels_ts)
# print 'predicted labels:', labels_ts
