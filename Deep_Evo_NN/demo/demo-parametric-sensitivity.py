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
----------------
Load UCI data sets
----------------
TODO: edit path to matlab files,
'''
# path = 'F:\Python\UCIDataset-matlab\UCI_25.mat'
path = 'F:\Python\Deep_dataset\cifar-10.mat'
mat = loadmat(path)
keys = mat.keys()
keys.remove('__version__')
keys.remove('__header__')
keys.remove('__globals__')
keys.sort()
save_name = 'result.npz'
# # load data
# keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo', 'optdigits']
p = Processor()
key = 'cifar'
data = mat[key]
X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')  # load_iris(return_X_y=True)
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
# # im_, gt_ = 'wuhanTM', 'wuhanTM_gt'
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
# if n_bands > 50:
#     pca_img = p.pca_transform(20, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, 20)
#     X, y = p.get_correct(pca_img, gt)
# else:
#     X, y = p.get_correct(img, gt)
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
-----------------------
Fashion MINST
-----------------------
'''
# path = 'F:\Python\Fashion-mnist-dataset\Fashion-mnist.mat'
# mat = sio.loadmat(path)
# X, y = mat['X'], mat['y'].flatten()
# X = PCA(n_components=20).fit_transform(X)
# # X, y = X[:1000], y[:1000]
# print('Fashion MINST:', X.shape, np.unique(y))


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

# hidden = [200, 200]
# clf = [DAE(hidden, ridge_alpha=C, logic_iter=max_iter, max_iter=max_iter),
#        ML_ELM(hidden.__len__(), hidden),
#        RidgeClassifier(alpha=C),
#        LogisticRegression(C=1e5, max_iter=max_iter),
#        BaseELM(hidden[-1]),
#        MLP(hidden, batch_size=32, learn_rate=.5, epochs=max_iter),
#        SAE(hidden, learn_rate=.5, epoch=max_iter, constraint=None),
#        SAE(hidden, learn_rate=.5, epoch=max_iter, constraint='L1')
#        # SAE(hidden, learn_rate=5, epoch=max_iter, constraint='L2')
#        ]
#
# key_clf = ['ours', 'ML-ELM', 'Ridge', 'LR', 'ELM', 'MLP', 'SAE', 'SAE-L1']
# for k, c in zip(key_clf, clf):
#     start = time.clock()
#     c.fit(X_train, y_train)
#     running_time = round(time.clock() - start, 3)
#     y_pre = c.predict(X_test)
#     print('%s : acc=%s, time=%s' % (k, accuracy_score(y_test, y_pre), running_time))
#
# # dae = DAE([10, 10, 10], max_iter=300)
# # dae.fit(X_train, y_train)
# # y_pre = dae.predict(X_test)
# # print('acc:', accuracy_score(y_test, y_pre))


hidden = range(10, 210, 10)
accs = []
t = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
print('train size: %s, train classes: %s' % (X_train.shape[0], np.unique(y_train)))
max_iter = 1000
C = 1e-6
for h in hidden:
    start = time.clock()
    dae = DAE([h]*2, ridge_alpha=C, logic_iter=max_iter, max_iter=max_iter)
    dae.fit(X_train, y_train)
    running_time = round(time.clock() - start, 3)
    y_pre = dae.predict(X_test)
    acc = np.round_(accuracy_score(y_test, y_pre) * 100, 2)
    accs.append(acc)
    t.append(running_time)
    print('----------------------------------------------')
    print('#hidden %s => acc=%s, time=%s' % (h, acc, running_time))

np.savez('Experiment/para_sensi-hidden-time-cifar-10.npz', acc=accs, time=t)


"-----------------------------------------------"
# import numpy as np
# import matplotlib.pyplot as plt
# p = 'F:\Python\Deep_Evo_NN\demo\Experiment\para_sensi-hidden-time-cifar-10.npz'
# a = np.load(p)
# acc, time = a['acc'], a['time']
# err = 100. - np.asanyarray(acc)
# hidden = range(10, 210, 10)
# fig, ax1 = plt.subplots()
# ax1.plot(hidden, err, 'b-')
# ax1.set_xlabel('Number Of Nodes')
# # Make the y-axis label, ticks and tick labels match the line color.
# ax1.set_ylabel('Test Error', color='b')
# ax1.tick_params('y', colors='b')
#
# ax2 = ax1.twinx()
# ax2.plot(hidden, time, 'r--')
# ax2.set_ylabel('Time Cost (Second)', color='r')
# ax2.tick_params('y', colors='r')
#
# fig.tight_layout()
# plt.grid(True)
# plt.show()
