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
# n_comp = 4
# n_row, n_clo, n_bands = img.shape
# # pca_img = p.pca_transform(n_comp, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, n_comp)
# X, y = p.get_correct(img, gt)

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
max_iter = 5000
ae_1 = EMO_AE_ELM(5, sparse_degree=0.05, max_iter=max_iter, n_pop=50)
ae_1.fit(X, X)
ae_1.save_evo_result('AE-ELM-Obj_L1')
X_1 = ae_1.predict(X)

ae_2 = EMO_AE_ELM(5, sparse_degree=0.001, max_iter=max_iter, n_pop=50)
ae_2.fit(X_1, X_1)
ae_2.save_evo_result('AE-ELM-Obj_L2')
X_2 = ae_2.predict(X_1)

'''
----------------------------------
random deep
'''
# # random mapping
def fea_extractor(mapping_matrix, X):
    X_ = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    map_X = expit(np.dot(X_, mapping_matrix))
    return map_X

# W_rand_1 = np.random.uniform(-1., 1., size=(X.shape[1] + 1, 2))
# X_1 = fea_extractor(W_rand_1, X)
#
# W_rand_2 = np.random.uniform(-1., 1., size=(X_1.shape[1] + 1, 3))
# X_2 = fea_extractor(W_rand_2, X_1)
#
# W_rand_3 = np.random.uniform(-1., 1., size=(X_2.shape[1] + 1, 5))
# X_3 = fea_extractor(W_rand_3, X_2)

# ae_3 = EMO_AE_ELM(50, sparse_degree=0.05, max_iter=max_iter*5, n_pop=50)
# ae_3.fit(X_2, X_2)
# ae_3.save_evo_result('AE-ELM-Obj')
# X_3 = ae_3.predict(X_2)

'''
-------------------------------------------------
Deep feature classification using ELM
'''
# n_hidden = 50
# accs = []
# for i in range(20):
#     train_index, test_index = p.get_tr_tx_index(y, test_size=0.4)
#     y_train, y_test = y[train_index], y[test_index]
#     elm_0_acc = accuracy_score(y_test, BaseELM(n_hidden).fit(X[train_index], y_train).predict(X[test_index]))
#     elm_1_acc = accuracy_score(y_test, BaseELM(n_hidden).fit(X_1[train_index], y_train).predict(X_1[test_index]))
#     elm_2_acc = accuracy_score(y_test, BaseELM(n_hidden).fit(X_2[train_index], y_train).predict(X_2[test_index]))
#     elm_3_acc = accuracy_score(y_test, BaseELM(n_hidden).fit(X_3[train_index], y_train).predict(X_3[test_index]))
#
#     accs.append([ elm_0_acc,elm_1_acc, elm_2_acc, elm_3_acc])
#     print('NO.', i, '----------------------------------------------')
#     print('\t\tX', '\t\tX_1', '\t\t\tX_2', '\t\t\tX_3')
#     print('  ', elm_0_acc, '     ', elm_1_acc, '     ', elm_2_acc, '      ', elm_3_acc)
#
# accs = np.asarray(accs)
# np.savez('deep_feature_test.npz', deep_fea=accs)
#
# print ('Raw:', np.round(accs.mean(axis=0) * 100, 2)[0], '+-', np.round(accs.std(axis=0) * 100, 2)[0], '  ', \
#        np.round(accs.mean(axis=0) * 100, 2)[1], '+-', np.round(accs.std(axis=0) * 100, 2)[1], '    ', \
#        np.round(accs.mean(axis=0) * 100, 2)[2], '+-', np.round(accs.std(axis=0) * 100, 2)[2], '    ', \
#        np.round(accs.mean(axis=0) * 100, 2)[3], '+-', np.round(accs.std(axis=0) * 100, 2)[3])


'''
------------------------------------------------
Deep feature classification using svm/dt/knn/nb
'''

X_raw = X
map_X_1 = X_1
map_X_2 = X_2
# map_X_3 = X_3
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
np.savez('deep_features.npz', X=X, X_1=X_1, X_2=X_2, X_pca=X_pca)
print ('Deep features have been saved in file deep_features.npz')
# map_X_3 = X_3

accs_raw = []
accs_X_1 = []
accs_X_2 = []
accs_X_3 = []
for i in range(10):
    train_index, test_index = p.get_tr_tx_index(y, test_size=0.4)

    X_train_raw, X_test_raw, y_train, y_test = X_raw[train_index], X_raw[test_index], y[train_index], y[test_index]
    X_1_train, X_1_test = map_X_1[train_index], map_X_1[test_index]
    X_2_train, X_2_test = map_X_2[train_index], map_X_2[test_index]
    # X_3_train, X_3_test = map_X_3[train_index], map_X_3[test_index]

    # svm_res_raw = accuracy_score(y_test, SVC(C=100, kernel='linear').fit(X_train_raw, y_train).predict(X_test_raw))
    parameters = {'C': [1, 10, 100, 10e3, 10e4, 10e5]}
    svm_res_raw = accuracy_score(y_test, GridSearchCV(SVC(), parameters).fit(X_train_raw, y_train).predict(X_test_raw))
    knn_res_raw = accuracy_score(y_test, KNeighborsClassifier().fit(X_train_raw, y_train).predict(X_test_raw))
    dt_res_raw = accuracy_score(y_test, DecisionTreeClassifier().fit(X_train_raw, y_train).predict(X_test_raw))
    nb_res_raw = accuracy_score(y_test, GaussianNB().fit(X_train_raw, y_train).predict(X_test_raw))

    svm_res_X1 = accuracy_score(y_test, GridSearchCV(SVC(), parameters).fit(X_1_train, y_train).predict(X_1_test))
    knn_res_X1 = accuracy_score(y_test, KNeighborsClassifier().fit(X_1_train, y_train).predict(X_1_test))
    dt_res_X1 = accuracy_score(y_test, DecisionTreeClassifier().fit(X_1_train, y_train).predict(X_1_test))
    nb_res_X1 = accuracy_score(y_test, GaussianNB().fit(X_1_train, y_train).predict(X_1_test))

    svm_res_X2 = accuracy_score(y_test, GridSearchCV(SVC(), parameters).fit(X_2_train, y_train).predict(X_2_test))
    knn_res_X2 = accuracy_score(y_test, KNeighborsClassifier().fit(X_2_train, y_train).predict(X_2_test))
    dt_res_X2 = accuracy_score(y_test, DecisionTreeClassifier().fit(X_2_train, y_train).predict(X_2_test))
    nb_res_X2 = accuracy_score(y_test, GaussianNB().fit(X_2_train, y_train).predict(X_2_test))

    # svm_res_X3 = accuracy_score(y_test, GridSearchCV(SVC(), parameters).fit(X_3_train, y_train).predict(X_3_test))
    # knn_res_X3 = accuracy_score(y_test, KNeighborsClassifier().fit(X_3_train, y_train).predict(X_3_test))
    # dt_res_X3 = accuracy_score(y_test, DecisionTreeClassifier().fit(X_3_train, y_train).predict(X_3_test))
    # nb_res_X3 = accuracy_score(y_test, GaussianNB().fit(X_3_train, y_train).predict(X_3_test))

    print ('NO.', i,  '----------------------------------------------')
    print('\t\tSVM', '\t\tkNN', '\t\t\tDT', '\t\t\tNB')
    print('     Raw:', svm_res_raw, '     ', knn_res_raw, '     ', dt_res_raw, '      ', nb_res_raw)
    print('     random:', svm_res_X1, '     ', knn_res_X1, '     ', dt_res_X1, '      ', nb_res_X1)
    print('     MOEA:', svm_res_X2, '     ', knn_res_X2, '     ', dt_res_X2, '      ', nb_res_X2)
    # print('     MOEA:', svm_res_X3, '     ', knn_res_X3, '     ', dt_res_X3, '      ', nb_res_X3)

    accs_raw.append([svm_res_raw, knn_res_raw, dt_res_raw, nb_res_raw])
    accs_X_1.append([svm_res_X1, knn_res_X1, dt_res_X1, nb_res_X1])
    accs_X_2.append([svm_res_X2, knn_res_X2, dt_res_X2, nb_res_X2])
    # accs_X_3.append([svm_res_X3, knn_res_X3, dt_res_X3, nb_res_X3])

accs_raw = np.asarray(accs_raw)
accs_X_1 = np.asarray(accs_X_1)
accs_X_2 = np.asarray(accs_X_2)
accs_X_3 = np.asarray(accs_X_3)

np.savez('deep_feature_accs.npz', X_raw=accs_raw, X_1=accs_X_1, X_2=accs_X_2, X_3=accs_X_3)

print ('Raw:', np.round(accs_raw.mean(axis=0) * 100, 2)[0], '+-', np.round(accs_raw.std(axis=0) * 100, 2)[0], '  ', \
       np.round(accs_raw.mean(axis=0) * 100, 2)[1], '+-', np.round(accs_raw.std(axis=0) * 100, 2)[1], '    ', \
       np.round(accs_raw.mean(axis=0) * 100, 2)[2], '+-', np.round(accs_raw.std(axis=0) * 100, 2)[2], '    ', \
       np.round(accs_raw.mean(axis=0) * 100, 2)[3], '+-', np.round(accs_raw.std(axis=0) * 100, 2)[3])

print ('X_1:', np.round(accs_X_1.mean(axis=0) * 100, 2)[0], '+-', np.round(accs_X_1.std(axis=0) * 100, 2)[0], '  ', \
       np.round(accs_X_1.mean(axis=0) * 100, 2)[1], '+-', np.round(accs_X_1.std(axis=0) * 100, 2)[1], '    ', \
       np.round(accs_X_1.mean(axis=0) * 100, 2)[2], '+-', np.round(accs_X_1.std(axis=0) * 100, 2)[2], '    ', \
       np.round(accs_X_1.mean(axis=0) * 100, 2)[3], '+-', np.round(accs_X_1.std(axis=0) * 100, 2)[3])

print ('X_2:', np.round(accs_X_2.mean(axis=0) * 100, 2)[0], '+-', np.round(accs_X_2.std(axis=0) * 100, 2)[0], '  ', \
       np.round(accs_X_2.mean(axis=0) * 100, 2)[1], '+-', np.round(accs_X_2.std(axis=0) * 100, 2)[1], '    ', \
       np.round(accs_X_2.mean(axis=0) * 100, 2)[2], '+-', np.round(accs_X_2.std(axis=0) * 100, 2)[2], '    ', \
       np.round(accs_X_2.mean(axis=0) * 100, 2)[3], '+-', np.round(accs_X_2.std(axis=0) * 100, 2)[3])

print ('X_3:', np.round(accs_X_3.mean(axis=0) * 100, 2)[0], '+-', np.round(accs_X_3.std(axis=0) * 100, 2)[0], '  ', \
       np.round(accs_X_3.mean(axis=0) * 100, 2)[1], '+-', np.round(accs_X_3.std(axis=0) * 100, 2)[1], '    ', \
       np.round(accs_X_3.mean(axis=0) * 100, 2)[2], '+-', np.round(accs_X_3.std(axis=0) * 100, 2)[2], '    ', \
       np.round(accs_X_3.mean(axis=0) * 100, 2)[3], '+-', np.round(accs_X_3.std(axis=0) * 100, 2)[3])

print ('--------------------------Plot--------------------------')
print('raw_mean, raw_std =', np.round(accs_raw.mean(axis=0) * 100, 2).tolist(), ',', np.round(accs_raw.std(axis=0) * 100, 2).tolist())
print('X_1_mean, X_1_std =', np.round(accs_X_1.mean(axis=0) * 100, 2).tolist(), ',', np.round(accs_X_1.std(axis=0) * 100, 2).tolist())
print('X_2_mean, X_2_std =', np.round(accs_X_2.mean(axis=0) * 100, 2).tolist(), ',', np.round(accs_X_2.std(axis=0) * 100, 2).tolist())
print('X_3_mean, X_3_std =', np.round(accs_X_3.mean(axis=0) * 100, 2).tolist(), ',', np.round(accs_X_3.std(axis=0) * 100, 2).tolist())


'''
----------------------------------------------------
Plot deep feature with histogram  
'''
# import matplotlib.pyplot as plt
# p = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\deep_features.npz'
# m2 = np.load(p)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(m2['X'].ravel(), bins=np.arange(0, 1., 0.01))


# p = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\AE-ELM-Obj_L2.npz'
# m1 = np.load(p)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlabel('$f_1$')
# ax.set_ylabel('$f_2$')
# ax.scatter(m1['obj'][:,0], m1['obj'][:,1])
