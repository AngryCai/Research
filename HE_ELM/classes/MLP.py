import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
from keras.optimizers import SGD, Adam


class MLP:

    def __init__(self, batch_size=32, epochs=500):
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        num_classes = np.unique(y).__len__()
        y_train = keras.utils.to_categorical(y, num_classes)
        model = Sequential()
        model.add(Dense(20, activation='sigmoid', input_shape=(X.shape[1],)))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(num_classes, activation='softmax'))
        # model.summary()
        sgd = SGD(lr=0.02, nesterov=True)
        model.compile(loss='categorical_crossentropy',
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
# elm = MLP(16, 1000)
# elm.fit(X_train, y_train)
# y_pre = elm.predict(X_test)
# print accuracy_score(y_test, y_pre)
# print y_pre