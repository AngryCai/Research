from __future__ import print_function
import sklearn.datasets as dt
from scipy.special._ufuncs import expit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from ELM import BaseELM
from ExtremeLearningMachine.HE_ELM.Kernel_ELM import KELM
from EMO_ELM import EMO_ELM, DE_ELM
from Deep_ELM import DeepELM
from scipy.io import loadmat
import numpy as np
from EMO_AE_ELM import EMO_AE_ELM, Autoencoder
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
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
results = {}
# # load data
# keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo']
p = Processor()
key = 'wdbc'
data = mat[key]
X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')

'''
__________________________________________________________________
Load HSI data
'''
# # gt_path = 'F:\Python\HSI_Files\Salinas_gt.mat'
# # img_path = 'F:\Python\HSI_Files\Salinas_corrected.mat'
#
gt_path = 'F:\Python\HSI_Files\Indian_pines_gt.mat'
img_path = 'F:\Python\HSI_Files\Indian_pines_corrected.mat'
#
# # gt_path = 'F:\Python\HSI_Files\PaviaU_gt.mat'
# # img_path = 'F:\Python\HSI_Files\PaviaU.mat'
#
# print(img_path)
# img, gt = p.prepare_data(img_path, gt_path)
# n_comp = 6
# n_row, n_clo, n_bands = img.shape
# pca_img = p.pca_transform(n_comp, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, n_comp)
# X, y = p.get_correct(pca_img, gt)
# X = MinMaxScaler().fit_transform(X)
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


'''
---------------------------------------------------------------
Accuracy comparison on 10 Algorithms
'''
n_hidden = 50
accs = []
for i in range(5):
    train_index, test_index = p.get_tr_tx_index(y, test_size=0.4)
    print ('# train = ', train_index.shape, '# test = ', test_index.shape)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    parameters = {'C': [1, 10, 100, 10e3, 10e4, 10e5]}
    svm_res = accuracy_score(y_test, GridSearchCV(SVC(), parameters).fit(X_train, y_train).predict(X_test))
    knn_res = accuracy_score(y_test, KNeighborsClassifier().fit(X_train, y_train).predict(X_test))
    dt_res = accuracy_score(y_test, DecisionTreeClassifier().fit(X_train, y_train).predict(X_test))
    nb_res = accuracy_score(y_test, GaussianNB().fit(X_train, y_train).predict(X_test))
    elm_res = accuracy_score(y_test, BaseELM(n_hidden).fit(X_train, y_train).predict(X_test))
    kelm_res = accuracy_score(y_test, KELM(C=10e3, kernel='linear').fit(X_train, y_train).predict(X_test))

    elm_ab = AdaBoostClassifier(BaseELM(n_hidden), algorithm="SAMME", n_estimators=500)
    elm_ab.fit(X_train, y_train)
    elm_ab_res = accuracy_score(y_test, elm_ab.predict(X_test))

    eelm_res = accuracy_score(y_test, DE_ELM(n_hidden, n_pop=50, max_iter=500).fit(X_train, y_train).predict(X_test))
    emo_elm_res = accuracy_score(y_test, EMO_ELM(n_hidden, sparse_degree=0.05, mu=0.3, max_iter=500, n_pop=50).fit(X_train, y_train).voting_predict(X_test))
    deep_elm_res = accuracy_score(y_test, DeepELM([50, 20], [500, 500], sparse_degree=[0.8, 0.1]).fit(X_train, y_train).predict(X_test))

    print ('NO.', i,  '----------------------------------------------')
    print('\t\tSVM', '\t\tkNN', '\t\t\tDT', '\t\t\tNB', '\t\t\tELM', '\t\t\tKELM', '\t\tAd-ELM', '\t\tE-ELM', '\t\t\tEMO-ELM', '\t\t\tDeep-ELM')
    print('     Raw:', svm_res, '     ', knn_res, '     ', dt_res, '      ', nb_res, '      ', elm_res, '      ', kelm_res, '      ',
          elm_ab_res, '      ', eelm_res, '      ', emo_elm_res,'      ', deep_elm_res)

    accs.append([svm_res, knn_res, dt_res, nb_res, elm_res, kelm_res, elm_ab_res, eelm_res, emo_elm_res, deep_elm_res])
    # accs_X_3.append([svm_res_X3, knn_res_X3, dt_res_X3, nb_res_X3])

accs = np.asarray(accs)
np.savez('deep_elm_accs.npz', acc=accs)

print ('Raw:', np.round(accs.mean(axis=0) * 100, 2)[0], '+-', np.round(accs.std(axis=0) * 100, 2)[0], '  ', \
       np.round(accs.mean(axis=0) * 100, 2)[1], '+-', np.round(accs.std(axis=0) * 100, 2)[1], '    ', \
       np.round(accs.mean(axis=0) * 100, 2)[2], '+-', np.round(accs.std(axis=0) * 100, 2)[2], '    ', \
       np.round(accs.mean(axis=0) * 100, 2)[3], '+-', np.round(accs.std(axis=0) * 100, 2)[3], '     ', \
       np.round(accs.mean(axis=0) * 100, 2)[4], '+-', np.round(accs.std(axis=0) * 100, 2)[4], '    ', \
       np.round(accs.mean(axis=0) * 100, 2)[5], '+-', np.round(accs.std(axis=0) * 100, 2)[5], '    ', \
       np.round(accs.mean(axis=0) * 100, 2)[6], '+-', np.round(accs.std(axis=0) * 100, 2)[6], '    ', \
       np.round(accs.mean(axis=0) * 100, 2)[7], '+-', np.round(accs.std(axis=0) * 100, 2)[7], '    ', \
       np.round(accs.mean(axis=0) * 100, 2)[8], '+-', np.round(accs.std(axis=0) * 100, 2)[8], '    ', \
       np.round(accs.mean(axis=0) * 100, 2)[9], '+-', np.round(accs.std(axis=0) * 100, 2)[9])
print ('-------------------------Plot format-----------------------')
print(accs)

