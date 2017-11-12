## test RP and multi-RP
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from ExtremeLearningMachine.MLRP.classes.MLRP import MLRP
import numpy as np
from scipy.io import loadmat
from Toolbox.Preprocessing import Processor



'''
__________________________________________________________________
Load UCI data sets
'''
path = 'F:\Python\UCIDataset-matlab\UCI_25.mat'
mat = loadmat(path)
keys = mat.keys()
keys.remove('__version__')
keys.remove('__header__')
keys.remove('__globals__')
keys.sort()
save_name = 'result.npz'
# # load data
# keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo']
p = Processor()
key = 'SPECTF'
data = mat[key]
X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')

''' 
___________________________________________________________________
Data pre-processing
'''
# remove these samples with small numbers
classes = np.unique(y)
print ('size:', X.shape, 'n_classes:', classes.shape[0])

for c in classes:
    if np.nonzero(y == c)[0].shape[0] < 10:
        X = np.delete(X, np.nonzero(y == c), axis=0)
        y = np.delete(y, np.nonzero(y == c))
y = p.standardize_label(y)
norm_X = MinMaxScaler().fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


n_hidden_1 = [50] * 50  # range(10, 1000, 10)#
n_hidden_2 = [10] * 50
n_hidden_3 = [20] * 50

# n_hidden_1.reverse()
# n_hidden_2.reverse()
XX_1 = MLRP(n_hidden_1).fit(X).predict(X)
print 'X1 is done'
XX_2 = MLRP(n_hidden_2).fit(X).predict(X)
print 'X2 is done'
XX_3 = MLRP(n_hidden_3).fit(X).predict(X)
print 'X3 is done'
# accuracy evaluation
clf = [RidgeClassifier(), LinearSVC()]  # LinearSVC() KNeighborsClassifier(n_neighbors=5)
cv = 3
accs_knn_1 = []
accs_svm_1 = []

accs_knn_2 = []
accs_svm_2 = []

accs_knn_3 = []
accs_svm_3 = []
for X_ in XX_1:
    accs_knn_1.append(cross_val_score(clf[0], X_, y, cv=cv))
    accs_svm_1.append(cross_val_score(clf[1], X_, y, cv=cv))

for X_ in XX_2:
    accs_knn_2.append(cross_val_score(clf[0], X_, y, cv=cv))
    accs_svm_2.append(cross_val_score(clf[1], X_, y, cv=cv))

for X_ in XX_3:
    accs_knn_3.append(cross_val_score(clf[0], X_, y, cv=cv))
    accs_svm_3.append(cross_val_score(clf[1], X_, y, cv=cv))

mean_knn_1 = np.asarray(accs_knn_1).mean(axis=1)
mean_svm_1 = np.asarray(accs_svm_1).mean(axis=1)
std_knn_1 = np.asarray(accs_knn_1).std(axis=1)
std_svm_1 = np.asarray(accs_svm_1).std(axis=1)

mean_knn_2 = np.asarray(accs_knn_2).mean(axis=1)
mean_svm_2 = np.asarray(accs_svm_2).mean(axis=1)
std_knn_2 = np.asarray(accs_knn_2).std(axis=1)
std_svm_2 = np.asarray(accs_svm_2).std(axis=1)

mean_knn_3 = np.asarray(accs_knn_3).mean(axis=1)
mean_svm_3 = np.asarray(accs_svm_3).mean(axis=1)
std_knn_3 = np.asarray(accs_knn_3).std(axis=1)
std_svm_3 = np.asarray(accs_svm_3).std(axis=1)

print 'knn:', mean_knn_1
# print 'std:', std_knn
print 'svm:', mean_svm_1
# print 'std:', std_svm

import matplotlib.pyplot as plt
# plt.plot(mean_knn)
ax = plt.subplot(111)
ax.plot(mean_knn_1, label='Ridge Linear ' + str(n_hidden_1[0]))
ax.plot(mean_knn_2, label='Ridge Linear ' + str(n_hidden_2[0]))
ax.plot(mean_knn_3, label='Ridge Linear ' + str(n_hidden_3[0]))
# ax.plot(mean_svm_1, label='Linear SVM 500')
# ax.plot(mean_svm_2, label='Linear SVM 50')
# ax.plot(mean_svm_3, label='Linear SVM 100')
ax.legend(loc='best')
plt.show()


# i1, i2, i3 = np.nonzero(y==0), np.nonzero(y==1), np.nonzero(y==2)
# plt.scatter(XX[7][i1][:, 0], XX[7][i1][:, 1])
# plt.scatter(XX[7][i2][:, 0], XX[7][i2][:, 1])
# plt.scatter(XX[7][i3][:, 0], XX[7][i3][:, 1])
print 'Done'

