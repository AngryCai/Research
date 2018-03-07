
from __future__ import print_function

import time

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Deep_Evo_NN.classes.ML_ELM import ML_ELM
from Deep_Evo_NN.classes.DeepAE import DAE
from Deep_Evo_NN.classes.StackedAE import SAE
from Toolbox.Preprocessing import Processor

'''
----------------
Load UCI data sets
----------------
TODO: edit path to matlab files,
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
# keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo', 'optdigits']
p = Processor()
key = 'optdigits'
data = mat[key]
X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')  # load_iris(return_X_y=True)
# X = PCA(n_components=20).fit_transform(X)


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
print('train size: %s, train classes: %s' % (X_train.shape[0], np.unique(y_train)))
max_iter = 1000
C = 1e-6
hidden = [200, 200]
clf = [
       DAE(hidden, ridge_alpha=C, logic_iter=max_iter, max_iter=max_iter),
       # SAE(hidden, learn_rate=.5, epoch=max_iter, constraint='L1')
       # ML_ELM(hidden.__len__(), hidden)
       ]

np.savez('./Experiment/X-y.npz', X=X_test, y=y_test)
key_clf = ['ours', 'SAE-L1']
for k, c in zip(key_clf, clf):
    start = time.clock()
    c.fit(X_train, y_train)
    running_time = round(time.clock() - start, 3)
    y_pre = c.predict(X_test, save_x_hat='./Experiment/' + k + '-X_hat.npz')
    c.save_model('./Experiment/' + k + '-weights.npz')
    print('%s : acc=%s, time=%s' % (k, accuracy_score(y_test, y_pre), running_time))

# dae = DAE([10, 10, 10], max_iter=300)
# dae.fit(X_train, y_train)
# y_pre = dae.predict(X_test)
# print('acc:', accuracy_score(y_test, y_pre))


# import matplotlib.pyplot as plt
# import numpy as np
# p_X_y = 'F:\Python\Deep_Evo_NN\demo\Experiment\X-y.npz'
# p_ours_X_hat = 'F:\Python\Deep_Evo_NN\demo\Experiment\ours-X_hat.npz'
# p_sae_X_hat = 'F:\Python\Deep_Evo_NN\demo\Experiment\SAE-L1-X_hat.npz'
#
# X_y, X_hat_ours, X_hat_sae = np.load(p_X_y), np.load(p_ours_X_hat)['X_hat'], np.load(p_sae_X_hat)['X_hat']
# X, y = X_y['X'], X_y['y']
#
# fig, axes = plt.subplots(nrows=3, ncols=2)
# for i in range(3):
#     index_i = np.nonzero(y == i)
#     axes[i, 0].matshow(X_hat_ours[0][index_i][:20], cmap=plt.cm.gray)
#     axes[i, 1].matshow(X_hat_sae[0][index_i][:20], cmap=plt.cm.gray)

# for i in [2, 4, 6]:
#     index_i = np.nonzero(y == i - 1)
#     plt.subplot(320 + i)
#     plt.matshow(X_hat_sae[index_i][:50])

print('None')


