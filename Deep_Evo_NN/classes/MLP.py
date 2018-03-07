import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
from keras.optimizers import SGD


class MLP:

    def __init__(self, layer_size, batch_size=24, learn_rate=0.1, epochs=1000):
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.epochs = epochs

    def fit(self, X, y):
        num_classes = np.unique(y).__len__()
        y_train = keras.utils.to_categorical(y, num_classes)
        model = Sequential()
        model.add(Dense(self.layer_size[0], activation='sigmoid', input_shape=(X.shape[1],)))
        for i in range(1, len(self.layer_size)):
            model.add(Dense(self.layer_size[i], activation='sigmoid'))
        model.add(Dense(num_classes, activation='sigmoid'))
        # model.summary()
        sgd = SGD(lr=self.learn_rate)
        model.compile(loss='mse',
                      optimizer=sgd,
                      metrics=['accuracy'])
        model.fit(X, y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose=0)
        self.model = model
        return self

    def predict(self, X):
        y_pre = self.model.predict(X)
        return y_pre.argmax(axis=1)


'''
---------
Test
---------
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
# lb = preprocessing.LabelBinarizer()
# Y_train = lb.fit_transform(y_train)
# # Y_test = lb.fit_transform(y_test)
#
# elm = MLP([10, 20], batch_size=10, epochs=1000)
# elm.fit(X_train, y_train)
# y_pre = elm.predict(X_test)
# print accuracy_score(y_test, y_pre)
# print y_pre