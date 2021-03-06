"""
Evaluate EMO-ELM parameters effect on performance
"""
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
from ELM import BaseELM
from ExtremeLearningMachine.CascadeELM.Kernel_ELM import KELM
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


'''
-------------------------------------------------------
mu/sparsity effect on accuracy 
'''
# keys = ['Iris','SPECTF', 'Wine','car','chart','cotton','dermatology','diabetes','ecoli','glass','letter','libras','optdigits',
# 'pen','sat','satellite','segment','soybean','vowel','wdbc','yeast','zoo']
# keys = ['Iris','Wine','wdbc','soybean','ecoli','diabetes']
# results = {}
# for key in keys:
#     print('processing ', key)
#     data = mat[key]
#     X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
#     # remove these samples with small numbers
#     # print X.shape, np.unique(y).shape[0]
#     classes = np.unique(y)
#     for c in classes:
#         if np.nonzero(y == c)[0].shape[0] < 10:
#             X = np.delete(X, np.nonzero(y == c), axis=0)
#             y = np.delete(y, np.nonzero(y == c))
#     p = Processor()
#     y = p.standardize_label(y)
#     X = StandardScaler().fit_transform(X)
#     # print 'num classes:', np.unique(y).__len__()
#     # run 10 times
#     test_score = []
#     n_hidden = 50
#     max_iter = 1000
#     n_pop = 50
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
#     Y_train = p.one2array(y_train)
#     print(X.shape[1], np.unique(y).shape[0], X_train.shape[0], X_test.shape[0])
#     sparsity_level = np.arange(0., 1.05, 0.1)
#     sparsity_level[0] = 0.001
#     sparsity_level[-1] = 0.98
#     mu = np.arange(0., 1.05, 0.1)
#     mu[0] = 0.001
#     mu[-1] = 0.98
#
#     for sp_ in sparsity_level:  # number of n_unit
#         for mu_ in mu:
#             print ('----------------------------------------')
#             print('sp=', sp_, '  mu=', mu_)
#             acc_round = []
#             for c in range(5):
#                 ours = EMO_ELM(n_hidden, sparse_degree=sp_, mu=mu_, max_iter=max_iter, n_pop=n_pop)
#                 ours.fit(X_train, y_train)
#                 label_ours = ours.voting_predict(X_test, y=y_test)
#                 acc_ours = accuracy_score(y_test, label_ours)
#                 acc_round.append(acc_ours)
#             test_score.append([np.round(np.asarray(acc_round).mean(axis=0) * 100, 2),
#                                np.round(np.asarray(acc_round).std(axis=0) * 100, 2)])
#             print('test error:', np.round(np.asarray(acc_round).mean(axis=0) * 100, 2))
#     results[key + '_acc'] = np.asarray(test_score)
# np.savez('sparsity_level_mu_effect.npz', results)


'''
-------------------------------------------------------
iteration effect on accuracy
ELM,KELM,Ada-ELM, E-ELM, EMO-ELM
'''
keys = ['Iris', 'Wine', 'wdbc', 'soybean', 'ecoli', 'diabetes']
results_mean = {}
results_std = {}

"""
for key in keys:
    print('processing ', key)
    data = mat[key]
    X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
    # remove these samples with small numbers
    # print X.shape, np.unique(y).shape[0]
    classes = np.unique(y)
    for c in classes:
        if np.nonzero(y == c)[0].shape[0] < 10:
            X = np.delete(X, np.nonzero(y == c), axis=0)
            y = np.delete(y, np.nonzero(y == c))
    p = Processor()
    y = p.standardize_label(y)
    X = StandardScaler().fit_transform(X)
    # print 'num classes:', np.unique(y).__len__()
    # run 10 times
    n_hidden = 10
    n_pop = 50
    C = 1000
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
    Y_train = p.one2array(y_train)
    print(X.shape[1], np.unique(y).shape[0], X_train.shape[0], X_test.shape[0])
    iteration = np.arange(0, 1001, 100)
    iteration[0] = 1
    total_acc_mean = []
    total_acc_std = []
    for i in iteration:
        print('-----------------------------------------------------------------------')
        print ('iter = ', i)
        acc = []
        for t in range(10):
            # Y_train = p.one2array(y_train)
            print(t, '-th round evaluating')
            '''Base ELM'''
            elm_base = BaseELM(n_hidden)
            elm_base.fit(X_train, y_train)
            y_pre_elmbase = elm_base.predict(X_test)
            acc_elmbase = accuracy_score(y_test, y_pre_elmbase)
            print('     ELM:', acc_elmbase)

            '''Kernel ELM'''
            kelm = KELM(C=C, kernel='rbf')
            kelm.fit(X_train, y_train)
            y_pre_kelm = kelm.predict(X_test)
            acc_kelm = accuracy_score(y_test, y_pre_kelm)
            print('     KELM:', acc_kelm)

            '''Adaboost ELM'''
            elm_ab = AdaBoostClassifier(BaseELM(n_hidden), algorithm="SAMME", n_estimators=i)
            elm_ab.fit(X_train, y_train)
            y_pre_elm_ab = elm_ab.predict(X_test)
            acc_elm_ab = accuracy_score(y_test, y_pre_elm_ab)
            print('     Ada-ELM:', acc_elm_ab)

            '''E-ELM'''
            melm = DE_ELM(n_hidden, n_pop=n_pop, max_iter=i)
            melm.fit(X_train, y_train)
            y_pre_melm = melm.predict(X_test)
            acc_de_elm = accuracy_score(y_test, y_pre_melm)
            print('     DE-ELM:', acc_de_elm)

            '''MO-ELM'''
            ours = EMO_ELM(n_hidden, sparse_degree=0.05, mu=0.3, max_iter=i, n_pop=n_pop)
            ours.fit(X_train, y_train)
            label_ours = ours.voting_predict(X_test, y=y_test)
            # label_ours = ours.best_predict(X_test)
            acc_ours = accuracy_score(y_test, label_ours)
            acc.append([acc_elmbase, acc_kelm, acc_elm_ab, acc_de_elm, acc_ours])
            print('     MO-ELM:', acc_ours)
        res = np.asarray(acc)
        total_acc_mean.append(np.round(res.mean(axis=0) * 100, 2))
        total_acc_std.append(np.round(res.std(axis=0) * 100, 2))
        print('\t\t\tELM', '\t\t\tKELM', '\t\t\tAdaBoost', '\t\t\tDE-ELM', '\t\t\tMO-ELM')
        print('AVG:', np.round(res.mean(axis=0) * 100, 2)[0], '+-', np.round(res.std(axis=0) * 100, 2)[0], '  ', \
                np.round(res.mean(axis=0) * 100, 2)[1], '+-', np.round(res.std(axis=0) * 100, 2)[1], '    ', \
                np.round(res.mean(axis=0) * 100, 2)[2], '+-', np.round(res.std(axis=0) * 100, 2)[2], '     ', \
                np.round(res.mean(axis=0) * 100, 2)[3], '+-', np.round(res.std(axis=0) * 100, 2)[3], '     ', \
                np.round(res.mean(axis=0) * 100, 2)[4], '+-', np.round(res.std(axis=0) * 100, 2)[4])
    results_mean[key + '_acc'] = np.asarray(total_acc_mean)
    results_std[key + '_acc'] = np.asarray(total_acc_std)
np.savez('iteration_effect_on_acc.npz', mean=np.asarray(results_mean), std=np.asarray(results_std))
"""

'''
-------------------------------------------------------
Sparse degree effect on accuracy
'''
"""
keys = ['Wine']#, 'Wine','wdbc','soybean','ecoli','diabetes']
results_mean = {}
results_std = {}
for key in keys:
    print('processing ', key)
    data = mat[key]
    X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
    # remove these samples with small numbers
    # print X.shape, np.unique(y).shape[0]
    classes = np.unique(y)
    for c in classes:
        if np.nonzero(y == c)[0].shape[0] < 10:
            X = np.delete(X, np.nonzero(y == c), axis=0)
            y = np.delete(y, np.nonzero(y == c))
    p = Processor()
    y = p.standardize_label(y)
    X = StandardScaler().fit_transform(X)
    # print 'num classes:', np.unique(y).__len__()
    # run 10 times
    n_hidden = 10
    n_pop = 50
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
    print(X.shape[1], np.unique(y).shape[0], X_train.shape[0], X_test.shape[0])
    rho = np.arange(0, 1., 0.05)
    rho[0] = 0.01
    total_acc_mean = []
    total_acc_std = []
    for i in rho:
        print('-----------------------------------------------------------------------')
        print ('sparse_level = ', i)
        acc = []
        for t in range(10):
            # Y_train = p.one2array(y_train)
            print(t, '-th round evaluating')
            ours = EMO_ELM(n_hidden, sparse_degree=i, mu=0.3, max_iter=500, n_pop=n_pop)
            ours.fit(X_train, y_train)
            label_ours = ours.voting_predict(X_test, y=y_test)
            # label_ours = ours.best_predict(X_test)
            acc_ours = accuracy_score(y_test, label_ours)
            acc.append(acc_ours)
            print('     MO-ELM:', acc_ours)
        res = np.asarray(acc)
        total_acc_mean.append(np.round(res.mean() * 100, 2))
        total_acc_std.append(np.round(res.std() * 100, 2))
        print('\t\t\tMO-ELM')
        print('AVG:', np.round(res.mean() * 100, 2), '+-', np.round(res.std() * 100, 2))
    results_mean[key + '_acc'] = np.asarray(total_acc_mean)
    results_std[key + '_acc'] = np.asarray(total_acc_std)
np.savez('sparse_degree_effect_on_acc.npz', mean=np.asarray(results_mean), std=np.asarray(results_std))
"""

'''
-------------------------------------------------------
For classification task: test iteration effect on acc/convergence 
'''
gen = [100, 500, 1000, 2000, 5000]
keys = ['ecoli']  #, 'Wine','wdbc','soybean','ecoli','diabetes']
results_mean = {}
results_std = {}
for key in keys:
    print('processing ', key)
    data = mat[key]
    X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
    # remove these samples with small numbers
    # print X.shape, np.unique(y).shape[0]
    classes = np.unique(y)
    for c in classes:
        if np.nonzero(y == c)[0].shape[0] < 10:
            X = np.delete(X, np.nonzero(y == c), axis=0)
            y = np.delete(y, np.nonzero(y == c))
    p = Processor()
    y = p.standardize_label(y)
    X = StandardScaler().fit_transform(X)
    # print 'num classes:', np.unique(y).__len__()
    # run 10 times
    n_hidden = 50
    n_pop = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
    print(X.shape[1], np.unique(y).shape[0], X_train.shape[0], X_test.shape[0])
    total_acc_mean = []
    total_acc_std = []
    for g in gen:
        print('-----------------------------------------------------------------------')
        print ('generation = ', g)
        acc = []
        for t in range(10):
            # Y_train = p.one2array(y_train)
            print(t, '-th round evaluating')
            ours = EMO_ELM(n_hidden, sparse_degree=0.05, mu=0.3, max_iter=g, n_pop=n_pop)
            ours.fit(X_train, y_train)
            if t == 4:
                ours.save_evo_result('convergence_acc_pareto_front_' + str(g))
            label_ours = ours.voting_predict(X_test, y=y_test)
            # label_ours = ours.best_predict(X_test)
            acc_ours = accuracy_score(y_test, label_ours)
            acc.append(acc_ours)
            print('     MO-ELM:', acc_ours)
        res = np.asarray(acc)
        total_acc_mean.append(np.round(res.mean() * 100, 2))
        total_acc_std.append(np.round(res.std() * 100, 2))
        print('\t\t\tMO-ELM')
        print('AVG:', np.round(res.mean() * 100, 2), '+-', np.round(res.std() * 100, 2))
    results_mean[key + '_acc'] = np.asarray(total_acc_mean)
    results_std[key + '_acc'] = np.asarray(total_acc_std)
np.savez('iteration_effect_on_acc.npz', mean=np.asarray(results_mean), std=np.asarray(results_std))





