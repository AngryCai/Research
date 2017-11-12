## test RP and multi-RP
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

from ExtremeLearningMachine.MLRP.classes.ELM import BaseELM
from ExtremeLearningMachine.MLRP.classes.MLRP import MLRP
import numpy as np
from scipy.io import loadmat

from ExtremeLearningMachine.MLRP.classes.MLRP_Classifier import MLRP_Classifier
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
# keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo']
p = Processor()
key = 'wdbc'
data = mat[key]
X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')


'''
__________________________________________________________________
# Load HSI data
# '''
# root = 'F:\\Python\\HSI_Files\\'
# # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
# # im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# im_, gt_ = 'KSC', 'KSC_gt'
#
# img_path = root + im_ + '.mat'
# gt_path = root + gt_ + '.mat'

#
# print(img_path)
# p = Processor()
# img, gt = p.prepare_data(img_path, gt_path)
# n_row, n_clo, n_bands = img.shape
# print ('img=', img.shape)
# # pca_img = p.pca_transform(n_comp, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, n_comp)
# X, y = p.get_correct(img, gt)
# print(X.shape)

'''
-----------------------
MINST
-----------------------
'''
# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata("MNIST original")
# p = Processor()
# # rescale the data, use the traditional train/test split
# X, y = mnist.data / 255., mnist.target
# print 'MINST:', X.shape, np.unique(y)
#

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


n = 1000
n_hidden = [n]  # range(50, 500, 50)#

acc = []
for dim in range(20):
    acc_ = cross_val_score(MLRP_Classifier(n_hidden*(dim+1)), X, y, cv=5)
    mean = np.asarray(acc_).mean()
    acc.append(mean)

np.savez('./Exp/dim_acc-.npz', acc=acc)

# clf = [
#     LinearSVC(),
#     KNeighborsClassifier(),
#     RidgeClassifier(),
#     BaseELM(n),
#     MLRP_Classifier(n_hidden_1)
#
# acc = []
# for c in clf:
#     temp_acc = cross_val_score(c, X, y, cv=10)
#     acc.append(np.asarray(temp_acc).mean())
#
# print acc


import matplotlib.pyplot as plt
a = np.load('./Exp/dim_acc-.npz')
acc = a['acc']
plt.plot(range(20), acc)
plt.show()
plt.savefig('./Exp/dim_acc-.eps', dpi=1000)
print 'Done'