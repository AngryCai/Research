"""
Implementation of AE with single hidden layer
"""
from __future__ import print_function
from keras.layers import Dense
from keras.models import Model
from keras.models import Sequential


class Autoencoder:

    def __init__(self, n_hidden, max_iter=100):
        self.n_hidden = n_hidden
        self.max_iter = max_iter

    def fit(self, X):
        model = Sequential()
        model.add(Dense(self.n_hidden, input_shape=(X.shape[1], ), activation='sigmoid', name='encode'))
        model.add(Dense(X.shape[1], activation='sigmoid'))
        model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
        model.fit(X, X, batch_size=16, epochs=self.max_iter,  verbose=0)
        self.model = model
        return self

    def predict(self, X):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer('encode').output)
        activations = intermediate_layer_model.predict(X)
        return activations