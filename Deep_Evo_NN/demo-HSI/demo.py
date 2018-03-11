## test RP and multi-RP
from __future__ import print_function
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from Deep_Evo_NN.classes.DeepAE import DAE
from Toolbox.Preprocessing import Processor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from Deep_Evo_NN.classes.MLP import MLP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from Deep_Evo_NN.classes.ELM import BaseELM
from Deep_Evo_NN.classes.StackedAE import SAE
from Deep_Evo_NN.classes.ML_ELM import ML_ELM
import time
import scipy.io as sio

'''
__________________________________________________________________
# Load HSI data
# '''
root = 'F:\\Python\\HSI_Files\\'
# im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
# im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
# im_, gt_ = 'Botswana', 'Botswana_gt'
# im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# im_, gt_ = 'KSC', 'KSC_gt'
im_, gt_ = 'wuhanTM', 'wuhanTM_gt'
#
img_path = root + im_ + '.mat'
gt_path = root + gt_ + '.mat'

print(img_path)
p = Processor()
img, gt = p.prepare_data(img_path, gt_path)
n_row, n_clo, n_bands = img.shape
print ('img=', img.shape)
if n_bands > 20:
    pca_img = p.pca_transform(20, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, 20)
    X, y = p.get_correct(pca_img, gt)
else:
    X, y = p.get_correct(img, gt)

print(X.shape)

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

X = MinMaxScaler().fit_transform(X)
n_clusters = np.unique(y).size
print(X.shape, np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
# X_train, X_test, y_train, y_test = p.split_each_class(X, y, each_train_size=1)
print('train size: %s, train classes: %s' % (X_train.shape[0], np.unique(y_train)))
max_iter = 2000
C = 1e-6
hidden = [200]
clf = [DAE(hidden, ridge_alpha=C, logic_iter=max_iter, max_iter=max_iter),
       ML_ELM(hidden.__len__(), hidden),
       RidgeClassifier(alpha=C),
       LogisticRegression(C=1e5, max_iter=max_iter),
       BaseELM(hidden[-1]),
       MLP(hidden, batch_size=12, learn_rate=.5, epochs=max_iter),
       SAE(hidden, learn_rate=.5, epoch=max_iter, constraint=None),
       SAE(hidden, learn_rate=.5, epoch=max_iter, constraint='L1')
       # SAE(hidden, learn_rate=5, epoch=max_iter, constraint='L2')
       ]

key_clf = ['ours', 'ML-ELM', 'Ridge', 'LR', 'ELM', 'MLP', 'SAE', 'SAE-L1']
for k, c in zip(key_clf, clf):
    start = time.clock()
    c.fit(X_train, y_train)
    running_time = round(time.clock() - start, 3)
    y_pre = c.predict(X_test)
    print('%s : acc=%s, time=%s' % (k, accuracy_score(y_test, y_pre), running_time))

# dae = DAE([10, 10, 10], max_iter=300)
# dae.fit(X_train, y_train)
# y_pre = dae.predict(X_test)
# print('acc:', accuracy_score(y_test, y_pre))


