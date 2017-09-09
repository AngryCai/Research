'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
from LASSO_ELM import LELM


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ExtremeLearningMachine.HE_ELM.ELM import BaseELM
from ExtremeLearningMachine.HE_ELM.MultiELM_ClassVector import DeepELM
from Toolbox.Preprocessing import Processor
import numpy as np

batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# y = Processor().standardize_label(y)
# convert class vectors to binary class matrices
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes)

elm_cv = LELM(500, C=1e-8, max_iter=5000)
elm_cv.fit(x_train, y_train)
y_pre_elmcv = elm_cv.predict(x_test)
acc_elmcv = accuracy_score(y_test, y_pre_elmcv)
print('Ours acc:', acc_elmcv)

elm_base = BaseELM(500)
elm_base.fit(x_train, y_train)
y_pre_elmbase = elm_base.predict(x_test)
acc_elmbase = accuracy_score(y_test, y_pre_elmbase)
print('ELM acc:', acc_elmbase)
