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

'''
----------------
Load UCI data sets
----------------
TODO: edit path to matlab files,
'''
path = r'F:\Python\UCIDataset-matlab\UCI_25.mat'
mat = loadmat(path)
# keys = mat.keys()
keys = list(mat)
keys.remove('__version__')
keys.remove('__header__')
keys.remove('__globals__')
keys.sort()
# load data
# keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo', 'optdigits']
key = 'Iris'
p = Processor()
data = mat[key]
X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
# X = PCA(n_components=20).fit_transform(X)


'''
__________________________________________________________________
# Load HSI data
# '''
# root = 'F:\\Python\\HSI_Files\\'
# # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
# # im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
# # im_, gt_ = 'Botswana', 'Botswana_gt'
# im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# # im_, gt_ = 'KSC', 'KSC_gt'
# #
# img_path = root + im_ + '.mat'
# gt_path = root + gt_ + '.mat'
#
# #
# print(img_path)
# p = Processor()
# img, gt = p.prepare_data(img_path, gt_path)
# n_row, n_clo, n_bands = img.shape
# print ('img=', img.shape)
# pca_img = p.pca_transform(20, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, 20)
# X, y = p.get_correct(pca_img, gt)
#
# print(X.shape)

'''
-----------------------
MINST
-----------------------
'''
# from sklearn.datasets import fetch_mldata
# from sklearn.decomposition import PCA
# mnist = fetch_mldata("MNIST original")
# # rescale the data, use the traditional train/test split
# X, y = mnist.data / 255., mnist.target
# X = PCA(n_components=20).fit_transform(X)
# # X, y = X[:1000], y[:1000]
# print('MINST:', X.shape, np.unique(y))


''' 
___________________________________________________________________
Data pre-processing
'''
# remove these samples with small numbers
classes = np.unique(y)
print ('size:', X.shape, 'n_classes:', classes.shape[0])

for c in classes:
    if np.nonzero(y == c)[0].shape[0] <= 10:
        X = np.delete(X, np.nonzero(y == c), axis=0)
        y = np.delete(y, np.nonzero(y == c))
y = p.standardize_label(y)

X = MinMaxScaler().fit_transform(X)
n_clusters = np.unique(y).size
print(X.shape, np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print('train size: %s, train classes: %s' % (X_train.shape[0], np.unique(y_train)))
max_iter = 1000
C = 1e-6
res = []
res_time = []
for i in range(1, 6):
    hidden = [50]*i
    clf = [DAE(hidden, ridge_alpha=C, logic_iter=max_iter, max_iter=max_iter),
           MLP(hidden, batch_size=32, learn_rate=0.5, epochs=max_iter),
           ML_ELM(hidden.__len__(), hidden),
           SAE(hidden, learn_rate=0.5, epoch=max_iter, constraint=None),
           SAE(hidden, learn_rate=0.5, epoch=max_iter, constraint='L1'),
           ]
    key_clf = ['ours', 'MLP', 'ML_ELM', 'SAE', 'SAE-L1']
    accs = []
    times = []
    for k, c in zip(key_clf, clf):
        start = time.clock()
        c.fit(X_train, y_train)
        running_time = round(time.clock() - start, 3)
        times.append(running_time)
        y_pre = c.predict(X_test)
        score = np.round(accuracy_score(y_test, y_pre) * 100, 2)
        print('%s : acc=%s, time=%s' % (k, score, running_time))
        accs.append(score)
    res.append(accs)
    res_time.append(times)

np.savez('Experiment/para-sensi-layer_Iris.npz', acc=np.asanyarray(res), time=np.asanyarray(res_time))

# dae = DAE([10, 10, 10], max_iter=300)
# dae.fit(X_train, y_train)
# y_pre = dae.predict(X_test)
# print('acc:', accuracy_score(y_test, y_pre))

# mlp = MLP(hidden, batch_size = 50, learn_rate = 0.0005, epochs=max_iter)
# mlp.fit(X_train, y_train)
# y_pre = mlp.predict(X_test)
# print ('acc:', accuracy_score(y_test, y_pre))


"===================Bar Plot========================="
# import numpy as np
import matplotlib.pyplot as plt
# p = 'F:\Python\Deep_Evo_NN\demo\Experiment\\para-sensi-layer_IndianPine.npz'
# npz = np.load(p)
# accs = np.asanyarray(npz['acc'])
# n_groups = 5
#
# ours_acc = accs[:, 0]
# mlelm_acc = accs[:, 2]
# sae_acc = accs[:, 4]
#
# fig, ax = plt.subplots()
# index = np.arange(n_groups)
# bar_width = 0.25
# opacity = 0.5
# rects1 = plt.bar(index - bar_width, ours_acc, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  label='Ours')
#
# rects2 = plt.bar(index, mlelm_acc, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  label='ML-ELM')
#
# rects3 = plt.bar(index + bar_width, sae_acc, bar_width,
#                  alpha=opacity,
#                  color='g',
#                  label=r'SAE-$L_1$')
#
# plt.xlabel('Number of Layer')
# plt.ylabel('Test Accuracy')
# plt.xticks(index, ('1', '2', '3', '4', '5'))
# plt.legend(loc=4)
#
#
# plt.tight_layout()
# plt.show()
plt.grid(axis=0)
