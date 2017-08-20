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
# keys=['soybean','Wine' ,'wdbc']
p = Processor()
key = 'Wine'
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
n_hidden = 3
for c in classes:
    if np.nonzero(y == c)[0].shape[0] < 10:
        X = np.delete(X, np.nonzero(y == c), axis=0)
        y = np.delete(y, np.nonzero(y == c))
y = p.standardize_label(y)
X = MinMaxScaler().fit_transform(X)

ae = EMO_AE_ELM(n_hidden, sparse_degree=0.05, max_iter=1000)
ae.fit(X, X)
ae.save_evo_result('AE-ELM-Obj')
print('The evolved results have been saved')
X_map = ae.predict(X)


'''_______________________________________________________________________
# Test mapping matrix using evolved W
'''
def fea_extractor(mapping_matrix, X):
    X_ = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    map_X = expit(np.dot(X_, mapping_matrix))
    return map_X
#
#
# # random mapping
W_rand = np.random.uniform(-1., 1., size=(X.shape[1] + 1, n_hidden))
map_X_rand = fea_extractor(W_rand, X)

# PCA
pca = PCA(n_components=n_hidden)
X_pca = pca.fit_transform(X)

# ae = Autoencoder(n_hidden, max_iter=5000)
# ae.fit(X)
# X_pca = ae.predict(X)

# MOEA-Mapping
map_X_moea = X_map
np.savez('fea_reduction3D.npz', rand=map_X_rand, pca=X_pca, moea=map_X_moea, y=y)

# pp = 'C:\Users\\07\OneDrive\ACADEMIC RESEARCH\Writing\EMO-ELM\Submission\Plot-raw-results-backup\HSI\Indian\\fea_reduction3D.npz'
# m = np.load(pp)
# map_X_moea, X_pca = m['moea'], m['pca']

accs_raw = []
accs_rand = []
accs_moea = []
for i in range(10):
    train_index, test_index = p.get_tr_tx_index(y, test_size=0.4)
    W_rand = np.random.uniform(-1., 1., size=(X.shape[1] + 1, n_hidden))
    map_X_rand = fea_extractor(W_rand, X)

    X_train_raw, X_test_raw, y_train, y_test = X_pca[train_index], X_pca[test_index], y[train_index], y[test_index]
    X_train_rand, X_test_rand = map_X_rand[train_index], map_X_rand[test_index]
    X_train_moea, X_test_moea = map_X_moea[train_index], map_X_moea[test_index]

    # X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(map_X_rand, y, train_size=0.6)
    # X_train_moea, X_test_moea, y_train_moea, y_test_moea = train_test_split(map_X_moea, y, train_size=0.6)

    # svm_res_raw = accuracy_score(y_test, SVC(C=100, kernel='linear').fit(X_train_raw, y_train).predict(X_test_raw))
    parameters = {'C': [1, 10, 100, 10e3, 10e4, 10e5]}
    svm_res_raw = accuracy_score(y_test, GridSearchCV(SVC(), parameters).fit(X_train_raw, y_train).predict(X_test_raw))
    knn_res_raw = accuracy_score(y_test, KNeighborsClassifier().fit(X_train_raw, y_train).predict(X_test_raw))
    dt_res_raw = accuracy_score(y_test, DecisionTreeClassifier().fit(X_train_raw, y_train).predict(X_test_raw))
    nb_res_raw = accuracy_score(y_test, GaussianNB().fit(X_train_raw, y_train).predict(X_test_raw))

    svm_res_rand = accuracy_score(y_test, GridSearchCV(SVC(), parameters).fit(X_train_rand, y_train).predict(X_test_rand))
    knn_res_rand = accuracy_score(y_test, KNeighborsClassifier().fit(X_train_rand, y_train).predict(X_test_rand))
    dt_res_rand = accuracy_score(y_test, DecisionTreeClassifier().fit(X_train_rand, y_train).predict(X_test_rand))
    nb_res_rand = accuracy_score(y_test, GaussianNB().fit(X_train_rand, y_train).predict(X_test_rand))

    svm_res_moea = accuracy_score(y_test, GridSearchCV(SVC(), parameters).fit(X_train_moea, y_train).predict(X_test_moea))
    knn_res_moea = accuracy_score(y_test, KNeighborsClassifier().fit(X_train_moea, y_train).predict(X_test_moea))
    dt_res_moea = accuracy_score(y_test, DecisionTreeClassifier().fit(X_train_moea, y_train).predict(X_test_moea))
    nb_res_moea = accuracy_score(y_test, GaussianNB().fit(X_train_moea, y_train).predict(X_test_moea))

    print ('NO.', i,  '----------------------------------------------')
    print('\t\tSVM', '\t\tkNN', '\t\t\tDT', '\t\t\tNB')
    print('     Raw:', svm_res_raw, '     ', knn_res_raw, '     ', dt_res_raw, '      ', nb_res_raw)
    print('     random:', svm_res_rand, '     ', knn_res_rand, '     ', dt_res_rand, '      ', nb_res_rand)
    print('     MOEA:', svm_res_moea, '     ', knn_res_moea, '     ', dt_res_moea, '      ', nb_res_moea)

    accs_raw.append([svm_res_raw, knn_res_raw, dt_res_raw, nb_res_raw])
    accs_rand.append([svm_res_rand, knn_res_rand, dt_res_rand, nb_res_rand])
    accs_moea.append([svm_res_moea, knn_res_moea, dt_res_moea, nb_res_moea])

accs_raw = np.asarray(accs_raw)
accs_rand = np.asarray(accs_rand)
accs_moea = np.asarray(accs_moea)
np.savez('pca_random_moea_acc.npz', pca_acc=accs_raw, rand_acc=accs_rand, emo_acc=accs_moea)

print ('Raw:', np.round(accs_raw.mean(axis=0) * 100, 2)[0], '+-', np.round(accs_raw.std(axis=0) * 100, 2)[0], '  ', \
       np.round(accs_raw.mean(axis=0) * 100, 2)[1], '+-', np.round(accs_raw.std(axis=0) * 100, 2)[1], '    ', \
       np.round(accs_raw.mean(axis=0) * 100, 2)[2], '+-', np.round(accs_raw.std(axis=0) * 100, 2)[2], '    ', \
       np.round(accs_raw.mean(axis=0) * 100, 2)[3], '+-', np.round(accs_raw.std(axis=0) * 100, 2)[3])

print ('Rand:', np.round(accs_rand.mean(axis=0) * 100, 2)[0], '+-', np.round(accs_rand.std(axis=0) * 100, 2)[0], '  ', \
       np.round(accs_rand.mean(axis=0) * 100, 2)[1], '+-', np.round(accs_rand.std(axis=0) * 100, 2)[1], '    ', \
       np.round(accs_rand.mean(axis=0) * 100, 2)[2], '+-', np.round(accs_rand.std(axis=0) * 100, 2)[2], '    ', \
       np.round(accs_rand.mean(axis=0) * 100, 2)[3], '+-', np.round(accs_rand.std(axis=0) * 100, 2)[3])

print ('MOEA:', np.round(accs_moea.mean(axis=0) * 100, 2)[0], '+-', np.round(accs_moea.std(axis=0) * 100, 2)[0], '  ', \
       np.round(accs_moea.mean(axis=0) * 100, 2)[1], '+-', np.round(accs_moea.std(axis=0) * 100, 2)[1], '    ', \
       np.round(accs_moea.mean(axis=0) * 100, 2)[2], '+-', np.round(accs_moea.std(axis=0) * 100, 2)[2], '    ', \
       np.round(accs_moea.mean(axis=0) * 100, 2)[3], '+-', np.round(accs_moea.std(axis=0) * 100, 2)[3])

print ('--------------------------Plot format--------------------------')
print('raw_mean, raw_std =', np.round(accs_raw.mean(axis=0) * 100, 2).tolist(), ',', np.round(accs_raw.std(axis=0) * 100, 2).tolist())
print('rand_mean, rand_std =', np.round(accs_rand.mean(axis=0) * 100, 2).tolist(), ',', np.round(accs_rand.std(axis=0) * 100, 2).tolist())
print('moea_mean, moea_std =', np.round(accs_moea.mean(axis=0) * 100, 2).tolist(), ',', np.round(accs_moea.std(axis=0) * 100, 2).tolist())
# for i in range(mu.__len__()):
#     print (i, '-th :', mu[i])
#     ours = MOEA_ELM(50, sparse_degree=mu[i], mu=0.3, max_iter=500)
#     ours.fit(X_train, y_train)
#     label_ours = ours.voting_predict(X_test, y=y_test)
#     # ours.save('./Exp/' + key +'_result_mu' + str(mu[i]))
#     print('acc:', accuracy_score(y_test, label_ours))
#     print ('----------------------------------------------------------------------------')



# #  example of plot codes
'''--------------------------------plot error bar with std-------------------------------------'''
##plot error bar with std
# import numpy as np
# import matplotlib.pyplot as plt
# 
# raw_mean, raw_std = [97.15, 95.35, 90.35, 97.71] , [1.46, 2.65, 3.66, 1.93]
# rand_mean, rand_std = [97.22, 95.0, 90.83, 96.67] , [1.46, 2.65, 3.66, 1.93]
# moea_mean, moea_std = [97.64, 95.9, 89.17, 96.04] , [1.59, 2.5, 3.39, 1.93]
# 
# ind = np.arange(4)  # the x locations for the groups
# width = 0.3       # the width of the bars
# 
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width, raw_mean, width, yerr=raw_std)
# rects2 = ax.bar(ind, rand_mean, width, yerr=rand_std)
# rects3 = ax.bar(ind + width, moea_mean, width, yerr=moea_std)
# 
# # add some text for labels, title and axes ticks
# ax.set_ylabel('Average testing accuracy (%)')
# # ax.set_title('Scores by group and gender')
# ax.set_xticks(ind)
# ax.set_xticklabels(('SVM', 'kNN', 'DT', 'NB'))
# ax.legend((rects1[0], rects2[0], rects3[0]), ('PCA', 'Random', 'EMO-ELM'), loc=4)
# 
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 0.86*height,
#                 '%.2f' % height,
#                 ha='center', va='bottom')
# 
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# 
# plt.show()

'''-----------------------------Plot Pareto Front--------------------------------------'''
## 
# p = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\AE-ELM-Obj.npz'
# m1 = np.load(p)
# plt.xlabel('$f_1$')
# plt.ylabel('$f_2$')
# plt.scatter(m1['obj'][:,0], m1['obj'][:,1])

'''-----------------------------Plot mapping--------------------------------------'''
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# pm = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\\fea_reduction3D.npz'
# 
# m3 = np.load(pm)
# y = m3['y']
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# s1 = ax.scatter(m3['pca'][np.nonzero(y == 0)][:,0], m3['pca'][np.nonzero(y == 0)][:,1], m3['pca'][np.nonzero(y == 0)][:,2], marker='o')
# s2 = ax.scatter(m3['pca'][np.nonzero(y == 1)][:,0], m3['pca'][np.nonzero(y == 1)][:,1], m3['pca'][np.nonzero(y == 1)][:,2], marker='^')
# s3 = ax.scatter(m3['pca'][np.nonzero(y == 2)][:,0], m3['pca'][np.nonzero(y == 2)][:,1], m3['pca'][np.nonzero(y == 2)][:,2], marker='s')
# ax.legend((s1, s2, s3), ('Setosa','Versicolour', 'Virginica'), loc=4)
# for angle in range(0, 60):
#     ax.view_init(30, angle)
#     plt.draw()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# s1 = ax.scatter(m3['moea'][np.nonzero(y == 0)][:,0], m3['moea'][np.nonzero(y == 0)][:,1], m3['moea'][np.nonzero(y == 0)][:,2], marker='o')
# s2 = ax.scatter(m3['moea'][np.nonzero(y == 1)][:,0], m3['moea'][np.nonzero(y == 1)][:,1], m3['moea'][np.nonzero(y == 1)][:,2], marker='^')
# s3 = ax.scatter(m3['moea'][np.nonzero(y == 2)][:,0], m3['moea'][np.nonzero(y == 2)][:,1], m3['moea'][np.nonzero(y == 2)][:,2], marker='s',)
# ax.legend((s1, s2, s3), ('Setosa','Versicolour', 'Virginica'), loc=4)
# 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# s1 = ax.scatter(m3['rand'][np.nonzero(y == 0)][:,0], m3['rand'][np.nonzero(y == 0)][:,1], m3['rand'][np.nonzero(y == 0)][:,2], marker='o')
# s2 = ax.scatter(m3['rand'][np.nonzero(y == 1)][:,0], m3['rand'][np.nonzero(y == 1)][:,1], m3['rand'][np.nonzero(y == 1)][:,2], marker='^')
# s3 = ax.scatter(m3['rand'][np.nonzero(y == 2)][:,0], m3['rand'][np.nonzero(y == 2)][:,1], m3['rand'][np.nonzero(y == 2)][:,2], marker='s')
# ax.legend((s1, s2, s3), ('Setosa','Versicolour', 'Virginica'), loc=4)


