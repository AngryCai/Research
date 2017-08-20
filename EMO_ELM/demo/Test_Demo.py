from __future__ import print_function
import sklearn.datasets as dt
from scipy.special._ufuncs import expit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from EMO_ELM.classes.ELM import BaseELM
from ExtremeLearningMachine.HE_ELM.Kernel_ELM import KELM
from EMO_ELM import EMO_ELM, DE_ELM
from scipy.io import loadmat
import numpy as np

from Toolbox.Preprocessing import Processor

path = 'F:\Python\UCIDataset-matlab\UCI_25.mat'
mat = loadmat(path)

keys = mat.keys()
keys.remove('__version__')
keys.remove('__header__')
keys.remove('__globals__')
keys.sort()
save_name = 'result.npz'
results = {}

# keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo']
#Iris SPECTF Wine abalone car chart cotton dermatology diabetes ecoli glass letter libras optdigits pen sat satellite
#segment shuttle soybean vowel wdbc yeast zoo

keys_haverun = ['Iris','Wine','cotton','wdbc','soybean','ecoli','chart','diabetes','glass','SPECTF',
                'abalone', 'shuttle', 'letter', 'pen', 'dermatology', 'car']
keys = ['glass']
for key in keys:
    print('processing ', key)
    # if key in keys_haverun:
    #     continue
    # # load data
    data = mat[key]
    X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
    # remove these samples with small numbers
    classes = np.unique(y)
    for c in classes:
        if np.nonzero(y == c)[0].shape[0] < 10:
            X = np.delete(X, np.nonzero(y == c), axis=0)
            y = np.delete(y, np.nonzero(y == c))
    p = Processor()
    y = p.standardize_label(y)
    X = StandardScaler().fit_transform(X)
    print ('num classes:', np.unique(y).__len__(), X.shape, 'train:test', int(y.shape[0]*0.6), y.shape[0] - int(y.shape[0]*0.6))

    # run 10 times
    accs = []
    n_hidden = 50  # int((X.shape[1] * np.unique(y).shape[0])**.5 + 10)
    max_iter = 500
    n_pop = 50
    C = 10e3
    for c in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
        # Y_train = p.one2array(y_train)
        print(c, '-th round evaluating')
        print ('-----------------------------------------------------------------------')
        '''Base ELM'''
        elm_base = BaseELM(n_hidden)
        elm_base.fit(X_train, y_train)
        y_pre_elmbase = elm_base.predict(X_test)
        acc_elmbase = accuracy_score(y_test, y_pre_elmbase)
        print('     ELM:', acc_elmbase)

        '''Kernel ELM'''
        kelm = KELM(C=C, kernel='linear')
        kelm.fit(X_train, y_train)
        y_pre_kelm = kelm.predict(X_test)
        acc_kelm = accuracy_score(y_test, y_pre_kelm)
        print('     KELM:', acc_kelm)

        '''Adaboost ELM'''
        elm_ab = AdaBoostClassifier(BaseELM(n_hidden), algorithm="SAMME", n_estimators=max_iter)
        elm_ab.fit(X_train, y_train)
        y_pre_elm_ab = elm_ab.predict(X_test)
        acc_elm_ab = accuracy_score(y_test, y_pre_elm_ab)
        print('     Ada-ELM:', acc_elm_ab)

        '''MultiELM-Feature'''
        melm = DE_ELM(n_hidden, n_pop=n_pop, max_iter=max_iter)
        melm.fit(X_train, y_train)
        y_pre_melm = melm.predict(X_test)
        acc_de_elm = accuracy_score(y_test, y_pre_melm)
        print('     DE-ELM:', acc_de_elm)

        '''MO-ELM'''
        ours = EMO_ELM(n_hidden, sparse_degree=0.2, mu=0.9, max_iter=max_iter, n_pop=n_pop)
        ours.fit(X_train, y_train)
        label_ours = ours.voting_predict(X_test, y=y_test)
        acc_ours = accuracy_score(y_test, label_ours)
        accs.append([acc_elmbase, acc_kelm, acc_elm_ab, acc_de_elm, acc_ours])
        print('     MO-ELM:', acc_ours)

    res = np.asarray(accs)
    results[key] = res

    print ('\t\t\tELM', '\t\t\tKELM', '\t\t\tAdaBoost', '\t\t\tDE-ELM', '\t\t\tMO-ELM')
    print ('AVG:', np.round(res.mean(axis=0)*100, 2)[0], '+-', np.round(res.std(axis=0)*100, 2)[0],'  ',\
        np.round(res.mean(axis=0)*100, 2)[1], '+-', np.round(res.std(axis=0)*100, 2)[1],'    ',\
        np.round(res.mean(axis=0) * 100, 2)[2], '+-', np.round(res.std(axis=0) * 100, 2)[2], '     ', \
        np.round(res.mean(axis=0) * 100, 2)[3], '+-', np.round(res.std(axis=0) * 100, 2)[3], '     ',\
        np.round(res.mean(axis=0) * 100, 2)[4], '+-', np.round(res.std(axis=0) * 100, 2)[4])
np.savez(save_name, results)


'''
----------------------------------------------
Statistic Mean +- Std
'''
# p = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\\result.npz'
# m = np.load(p)
# for k in m['arr_0'][()].keys():
#     print ('-----------------------------------------------------------')
#     print (k)
#     res = m['arr_0'][()][k]
#     print('\t\t\tELM', '\t\t\tKELM', '\t\t\tAdaBoost', '\t\t\tDE-ELM', '\t\t\tMO-ELM')
#     print('AVG:', np.round(res.mean(axis=0) * 100, 2)[0], '+-', np.round(res.std(axis=0) * 100, 2)[0], '  ', \
#           np.round(res.mean(axis=0) * 100, 2)[1], '+-', np.round(res.std(axis=0) * 100, 2)[1], '    ', \
#           np.round(res.mean(axis=0) * 100, 2)[2], '+-', np.round(res.std(axis=0) * 100, 2)[2], '     ', \
#           np.round(res.mean(axis=0) * 100, 2)[3], '+-', np.round(res.std(axis=0) * 100, 2)[3], '     ', \
#           np.round(res.mean(axis=0) * 100, 2)[4], '+-', np.round(res.std(axis=0) * 100, 2)[4])


