"""
Implementation of sparse auto encoder(SAE) with single hidden layer
-----------
loss = J(W,b) + beta*sum(KL(p||p_hat))
"""

from __future__ import print_function
from keras.layers import Dense
from keras.models import Model
from keras.models import Sequential
import theano.tensor as T
from keras import backend as K
from keras import regularizers
# from keras.regularizers import l2, ActivityRegularizer

class SAE:

    def __init__(self, n_hidden, max_iter=100):
        self.n_hidden = n_hidden
        self.max_iter = max_iter

    def fit(self, X):
        model = Sequential()
        model.add(Dense(self.n_hidden, input_shape=(X.shape[1], ), activation='sigmoid', name='encode',
                        activity_regularizer=regularizers.l1(l=1e-8)))  # self.kl_regularizer #regularizers.l1(0.01)
        model.add(Dense(X.shape[1], activation='sigmoid'))
        model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
        model.fit(X, X, batch_size=16, epochs=self.max_iter,  verbose=1)
        self.model = model
        return self

    def predict(self, X):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer('encode').output)
        activations = intermediate_layer_model.predict(X)
        return activations

    def __kl_divergence(self, p, p_hat):
        return (p * K.log(p / p_hat)) + ((1-p) * K.log((1-p) / (1-p_hat)))

    def kl_regularizer(self, weight_matrix):
        p_hat = K.sum(weight_matrix, axis=0)
        loss = 0.01 * self.__kl_divergence(0.05, p_hat)
        return 0.01 * K.sum(loss)

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import normalize
# X, y = load_iris(return_X_y=True)
# X = normalize(X)
#
# elm_ae = SAE(20, 100)
# elm_ae.fit(X)
# X_transform = elm_ae.predict(X)
# X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.4, random_state=42, stratify=y)
#
# from sklearn.neighbors import KNeighborsClassifier as KNN
# knn_1 = KNN(3)
# knn_1.fit(X_train, y_train)
# y_1 = knn_1.predict(X_test)
#
# # knn_2 = KNN(3)
# # knn_2.fit(X_train, y_train)
# # y_2 = knn_2.predict(X_test)
# acc = accuracy_score(y_test, y_1)
# print(acc)
