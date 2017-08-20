from sklearn import grid_search, preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from ELM_Layer import Layer, MultiELM
from ExtremeLearningMachine.HE_ELM.ELM import BaseELM
from Kernel_ELM import KELM
import scipy.io as sio
from Toolbox.Preprocessing import Processor

path = 'F:\Python\UCIDataset-matlab\UCI_25.mat'
mat = sio.loadmat(path)

keys = mat.keys()
keys.remove('__version__')
keys.remove('__header__')
keys.remove('__globals__')
keys.sort()
save_name = 'result.npz'
results = {}


''' 
----------------------
algorithm comparision
'''
# keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo']
#Iris SPECTF Wine abalone car chart cotton dermatology diabetes ecoli glass letter libras optdigits pen sat satellite
#segment shuttle soybean vowel wdbc yeast zoo
# keys = ['Iris', 'shuttle']#['optdigits', 'pen', 'sat', 'satellite', 'segment', 'shuttle', 'letter']
for key in keys:
    print 'processing ', key
    if key == 'abalone':# or key == 'letter' or key == 'pen' or key == 'shuttle':
        continue
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
    print 'num classes:', np.unique(y).__len__(), X.shape
    # run 10 times
    accs = []
    n_hidden = 50  # int((X.shape[1] * np.unique(y).shape[0])**.5 + 10)
    n_unit = 100
    C = 10e3
    for c in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
        Y_train = p.one2array(y_train)
        '''MultiELM-Feature'''
        melm = MultiELM(BaseELM(n_hidden, dropout_prob=0.5), KELM(C=C, kernel='rbf'), n_unit=n_unit)
        melm.fit(X_train, Y_train)
        y_pre_melm = melm.predict(X_test)
        acc_melm = accuracy_score(y_test, y_pre_melm)

        '''Base ELM'''
        elm_base = BaseELM(n_hidden)
        elm_base.fit(X_train, y_train)
        y_pre_elmbase = elm_base.predict(X_test)
        acc_elmbase = accuracy_score(y_test, y_pre_elmbase)
        print acc_melm, acc_elmbase

        '''Kernel ELM'''
        kelm = KELM(C=C, kernel='rbf')
        kelm.fit(X_train, y_train)
        y_pre_kelm = kelm.predict(X_test)
        acc_kelm = accuracy_score(y_test, y_pre_kelm)

        '''Adaboost ELM'''
        elm_ab = AdaBoostClassifier(BaseELM(n_hidden), algorithm="SAMME", n_estimators=n_unit)
        elm_ab.fit(X_train, y_train)
        y_pre_elm_ab = elm_ab.predict(X_test)
        acc_elm_ab = accuracy_score(y_test, y_pre_elm_ab)

        # print c, ' Ours ',acc_ours
        # print c, ' BaseELM ', acc_elmbase
        # print c, ' KELM', acc_kelm
        accs.append([acc_melm, acc_elmbase, acc_kelm, acc_elm_ab])
    res = np.asarray(accs)
    results[key] = res

    print '\tMELM\t', '\tELM\t', '\tKELM\t', '\tAdaBoost\t'
    print 'AVG:', np.round(res.mean(axis=0)*100, 2)[0], '+-', np.round(res.std(axis=0)*100, 2)[0],'  ',\
        np.round(res.mean(axis=0)*100, 2)[1], '+-', np.round(res.std(axis=0)*100, 2)[1],'  ',\
        np.round(res.mean(axis=0) * 100, 2)[2], '+-', np.round(res.std(axis=0) * 100, 2)[2], '  ', \
        np.round(res.mean(axis=0) * 100, 2)[3], '+-', np.round(res.std(axis=0) * 100, 2)[3]
        # np.round(res.mean(axis=0) * 100, 2)[4], '+-', np.round(res.std(axis=0) * 100, 2)[4]
np.savez(save_name, results)


''' 
-----------------
parameter test n_units
#  '''
# keys = ['Iris'] #['cotton', 'soybean', 'vowel', 'wdbc', 'yeast', 'zoo'] #['car']#, 'Iris', 'Wine',
# results = {}
# for key in keys:
#     print 'processing ', key
#     data = mat[key]
#     X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
#     # remove these samples with small numbers
#     print X.shape, np.unique(y).shape[0]
#     classes = np.unique(y)
#     for c in classes:
#         if np.nonzero(y == c)[0].shape[0] < 10:
#             X = np.delete(X, np.nonzero(y == c), axis=0)
#             y = np.delete(y, np.nonzero(y == c))
#     p = Processor()
#     y = p.standardize_label(y)
#     X = StandardScaler().fit_transform(X)
#     print 'num classes:', np.unique(y).__len__()
#     # run 10 times
#     train_score, test_score = [], []
#     train_score_ad, test_score_ad = [], []
#     n_hidden = 100
#     C = 1000
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
#     Y_train = p.one2array(y_train)
#     units = range(5, 101, 5)
#     units.insert(0, 1)
#     for n_unit in units:  # number of n_unit
#         print n_unit
#         acc_unit_tr, acc_unit_ts = [], []
#         acc_unit_tr_ad, acc_unit_ts_ad = [], []
#         for c in range(50):
#             '''MultiELM-Feature'''
#             melm = MultiELM(BaseELM(n_hidden, dropout_prob=0.98), KELM(C=C, kernel='rbf'), n_unit=n_unit)#KELM(C=C, kernel='rbf')
#             melm.fit(X_train, Y_train)
#             train_pre = melm.predict(X_train)
#             test_pre = melm.predict(X_test)
#             train_err = 1 - accuracy_score(y_train, train_pre)
#             test_err = 1 - accuracy_score(y_test, test_pre)
#             acc_unit_tr.append(train_err)
#             acc_unit_ts.append(test_err)
#
#             '''Adaboost ELM'''
#             elm_ab = AdaBoostClassifier(BaseELM(n_hidden), algorithm="SAMME", n_estimators=n_unit)
#             elm_ab.fit(X_train, y_train)
#             train_pre_ad = elm_ab.predict(X_train)
#             test_pre_ad = elm_ab.predict(X_test)
#             train_err_ad = 1 - accuracy_score(y_train, train_pre_ad)
#             test_err_ad = 1 - accuracy_score(y_test, test_pre_ad)
#             acc_unit_tr_ad.append(train_err_ad)
#             acc_unit_ts_ad.append(test_err_ad)
#
#         train_score.append([np.round(np.asarray(acc_unit_tr).mean(axis=0)*100, 2),
#                            np.round(np.asarray(acc_unit_tr).std(axis=0)*100, 2)])
#         test_score.append([np.round(np.asarray(acc_unit_ts).mean(axis=0) * 100, 2),
#                            np.round(np.asarray(acc_unit_ts).std(axis=0) * 100, 2)])
#         train_score_ad.append([np.round(np.asarray(acc_unit_tr_ad).mean(axis=0) * 100, 2),
#                             np.round(np.asarray(acc_unit_tr_ad).std(axis=0) * 100, 2)])
#         test_score_ad.append([np.round(np.asarray(acc_unit_ts_ad).mean(axis=0) * 100, 2),
#                            np.round(np.asarray(acc_unit_ts_ad).std(axis=0) * 100, 2)])
#         print 'train error:', np.round(np.asarray(acc_unit_tr).mean(axis=0) * 100, 2),\
#             'test error:', np.round(np.asarray(acc_unit_ts).mean(axis=0) * 100, 2)
#
#     results[key + '_train_err_ours'] = np.asarray(train_score)
#     results[key + '_test_err_ours'] = np.asarray(test_score)
#     results[key + '_train_err_adaboost'] = np.asarray(train_score_ad)
#     results[key + '_test_err_adaboost'] = np.asarray(test_score_ad)
# np.savez('n_unit_test_100_hidden.npz', results)

'''
--------------
comparision KEML/BaseELM/Ours in changed n_hidden
'''
# keys = ['Iris', 'soybean', 'vowel', 'wdbc', 'yeast', 'zoo', 'Wine', 'car', 'cotton']
# results = {}
# for key in keys:
#     print 'processing ', key
#     data = mat[key]
#     X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
#     # remove these samples with small numbers
#     classes = np.unique(y)
#     for c in classes:
#         if np.nonzero(y == c)[0].shape[0] < 10:
#             X = np.delete(X, np.nonzero(y == c), axis=0)
#             y = np.delete(y, np.nonzero(y == c))
#     p = Processor()
#     y = p.standardize_label(y)
#     X = StandardScaler().fit_transform(X)
#     print 'num classes:', np.unique(y).__len__()
#     # run 10 times
#     acc_ours = []
#     acc_base = []
#     acc_adaboost = []
#
#     C = 1000
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
#     Y_train = p.one2array(y_train)
#     hiddens = range(10, 201, 10)
#     hiddens.insert(0, 5)
#     for n_hidden in hiddens:  # number of n_unit
#         acc_unit_ours = []
#         acc_unit_base = []
#         acc_unit_tradaboost = []
#         for c in range(50):
#             '''MultiELM-Feature'''
#             melm = MultiELM(BaseELM(n_hidden, dropout_prob=0.5), KELM(C=C, kernel='rbf'), n_unit=100)
#             melm.fit(X_train, Y_train)
#             our_pre = melm.predict(X_test)
#             acc_ours_ = 1 - accuracy_score(y_test, our_pre)
#
#             '''Base ELM'''
#             elm_base = BaseELM(n_hidden)
#             elm_base.fit(X_train, y_train)
#             y_pre_elmbase = elm_base.predict(X_test)
#             acc_elmbase_ = 1 - accuracy_score(y_test, y_pre_elmbase)
#
#             # '''Kernel ELM'''
#             # kelm = KELM(C=C)
#             # kelm.fit(X_train, y_train)
#             # y_pre_kelm = kelm.predict(X_test)
#             # acc_kelm = 1 - accuracy_score(y_test, y_pre_kelm)
#
#             elm_ab = AdaBoostClassifier(BaseELM(n_hidden), algorithm="SAMME", n_estimators=100)
#             elm_ab.fit(X_train, y_train)
#             y_pre_elm_ab = elm_ab.predict(X_test)
#             acc_elm_ab_ = 1 - accuracy_score(y_test, y_pre_elm_ab)
#
#             acc_unit_ours.append(acc_ours_)
#             acc_unit_base.append(acc_elmbase_)
#             acc_unit_tradaboost.append(acc_elm_ab_)
#
#         acc_ours.append([np.round(np.asarray(acc_unit_ours).mean(axis=0)*100, 2), np.round(np.asarray(acc_unit_ours).std(axis=0)*100, 2)])
#         acc_base.append([np.round(np.asarray(acc_unit_base).mean(axis=0) * 100, 2), np.round(np.asarray(acc_unit_base).std(axis=0) * 100, 2)])
#         acc_adaboost.append([np.round(np.asarray(acc_unit_tradaboost).mean(axis=0) * 100, 2), np.round(np.asarray(acc_unit_tradaboost).std(axis=0) * 100, 2)])
#
#     results[key + '_ours_err'] = np.asarray(acc_ours)
#     results[key + '_base_err'] = np.asarray(acc_base)
#     results[key + '_adboost_err'] = np.asarray(acc_adaboost)
# np.savez('n_hidden_comparision_20_units.npz', results)


'''
-------------------
comparision KEML/BaseELM/KEML/Ours in changed train percentage
'''

# keys = ['wdbc']#['vowel', 'wdbc', 'yeast', 'zoo', 'Wine', 'car', 'cotton','Iris','soybean']
#
# results = {}
# for key in keys:
#     print 'processing ', key
#     data = mat[key]
#     X, y = data[:, 1:].astype('float32'), data[:, 0].astype('int8')
#     # remove these samples with small numbers
#     classes = np.unique(y)
#     for c in classes:
#         if np.nonzero(y == c)[0].shape[0] < 10:
#             X = np.delete(X, np.nonzero(y == c), axis=0)
#             y = np.delete(y, np.nonzero(y == c))
#     p = Processor()
#     y = p.standardize_label(y)
#     X = StandardScaler().fit_transform(X)
#     # run 10 times
#     acc_ours = []
#     acc_base = []
#     acc_adaboost = []
#     acc_keml = []
#     C = 10e3
#     n_hidden = 100 #int((X.shape[1] * np.unique(y).__len__()) ** .5 + 10)
#     tr_index, ts_index = p.get_tr_tx_index(y, test_size=0.4)
#     print 'num classes:', np.unique(y).__len__(), 'n_hidden=', n_hidden
#     percentage = np.arange(0.1, 1.1, 0.1)
#     X_test, y_test = X[ts_index], y[ts_index]
#     for per in percentage:  # number of n_unit
#         acc_unit_ours = []
#         acc_unit_base = []
#         acc_unit_tradaboost = []
#         acc_unit_keml = []
#         # tr_index, ts_index = p.get_tr_tx_index(y, test_size=0.4)
#         # X_test, y_test = X[ts_index], y[ts_index]
#         # print 'num classes:', np.unique(y).__len__(), 'n_hidden=', n_hidden
#         # X_train,  _, y_train, _ = train_test_split(X, y, train_size=0.6 * per, stratify=y)#X[tr_index][:top], y[tr_index][:top]
#         # print X_train.shape, '\n',  np.unique(y_train)
#         # Y_train = p.one2array(y_train)
#         for c in range(50):
#             '''MultiELM-Feature'''
#             if per == 1:
#                 X_train, y_train = X[tr_index], y[tr_index]
#             else:
#                 X_train, _, y_train, _ = train_test_split(X[tr_index], y[tr_index], train_size=per,
#                                                       stratify=y[tr_index])
#             # print X_train.shape, '\n', np.unique(y_train)
#             Y_train = p.one2array(y_train)
#             melm = MultiELM(BaseELM(n_hidden, dropout_prob=0.9), KELM(C=C, kernel='rbf'), n_unit=100)
#             melm.fit(X_train, Y_train)
#             our_pre = melm.predict(X_test)
#             acc_ours_ = 1 - accuracy_score(y_test, our_pre)
#
#             '''Base ELM'''
#             elm_base = BaseELM(n_hidden)
#             elm_base.fit(X_train, y_train)
#             y_pre_elmbase = elm_base.predict(X_test)
#             acc_elmbase_ = 1 - accuracy_score(y_test, y_pre_elmbase)
#
#             '''Kernel ELM'''
#             kelm = KELM(C=C)
#             kelm.fit(X_train, y_train)
#             y_pre_kelm = kelm.predict(X_test)
#             acc_kelm_ = 1 - accuracy_score(y_test, y_pre_kelm)
#
#             elm_ab = AdaBoostClassifier(BaseELM(n_hidden), algorithm="SAMME", n_estimators=100)
#             elm_ab.fit(X_train, y_train)
#             y_pre_elm_ab = elm_ab.predict(X_test)
#             acc_elm_ab_ = 1 - accuracy_score(y_test, y_pre_elm_ab)
#
#             acc_unit_ours.append(acc_ours_)
#             acc_unit_base.append(acc_elmbase_)
#             acc_unit_tradaboost.append(acc_elm_ab_)
#             acc_unit_keml.append(acc_kelm_)
#
#         acc_ours.append([np.round(np.asarray(acc_unit_ours).mean(axis=0)*100, 2), np.round(np.asarray(acc_unit_ours).std(axis=0)*100, 2)])
#         acc_base.append([np.round(np.asarray(acc_unit_base).mean(axis=0) * 100, 2), np.round(np.asarray(acc_unit_base).std(axis=0) * 100, 2)])
#         acc_adaboost.append([np.round(np.asarray(acc_unit_tradaboost).mean(axis=0) * 100, 2), np.round(np.asarray(acc_unit_tradaboost).std(axis=0) * 100, 2)])
#         acc_keml.append([np.round(np.asarray(acc_unit_keml).mean(axis=0) * 100, 2), np.round(np.asarray(acc_unit_keml).std(axis=0) * 100, 2)])
#
#     results[key + '_ours_err'] = np.asarray(acc_ours)
#     results[key + '_base_err'] = np.asarray(acc_base)
#     results[key + '_adboost_err'] = np.asarray(acc_adaboost)
#     results[key + '_kelm_err'] = np.asarray(acc_keml)
# np.savez('n_training_samples_comparision_50_hidden.npz', results)




''' 
-----------------
parameter test n_units + droprate
#  '''
# keys = ['Iris','SPECTF', 'Wine','car','chart','cotton','dermatology','diabetes','ecoli','glass','letter','libras','optdigits',
# 'pen','sat','satellite','segment','soybean','vowel','wdbc','yeast','zoo']
# results = {}
# for key in keys:
#     print 'processing ', key
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
#     train_score, test_score = [], []
#     train_score_ad, test_score_ad = [], []
#     n_hidden = 100
#     C = 10e3
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
#     Y_train = p.one2array(y_train)
#
#     print X.shape[1], np.unique(y).shape[0], X_train.shape[0], X_test.shape[0]
#     #grid search
#     parameters = {'C': [1, 10, 10e2, 10e3, 10e4, 10e5]}
#     # clf = grid_search.GridSearchCV(KELM(kernel='rbf'), parameters, cv=3)
#     # clf.fit(X_train, y_train)
#     # C = clf.best_params_['C']
#     # print 'best C:', C
#     units = range(5, 101, 5)
#     units.insert(0, 1)
#     drop_rate = np.arange(0., 1, 0.05)
#     for n_unit in units:  # number of n_unit
#         for drop in drop_rate:
#             print n_unit, drop
#             acc_unit_tr, acc_unit_ts = [], []
#             for c in range(50):
#                 '''MultiELM-Feature'''
#                 melm = MultiELM(BaseELM(n_hidden, dropout_prob=drop), KELM(C=C, kernel='rbf'), n_unit=n_unit)
#                 melm.fit(X_train, Y_train)
#                 train_pre = melm.predict(X_train)
#                 test_pre = melm.predict(X_test)
#                 train_err = 1 - accuracy_score(y_train, train_pre)
#                 test_err = 1 - accuracy_score(y_test, test_pre)
#                 acc_unit_tr.append(train_err)
#                 acc_unit_ts.append(test_err)
#             train_score.append([np.round(np.asarray(acc_unit_tr).mean(axis=0)*100, 2),
#                            np.round(np.asarray(acc_unit_tr).std(axis=0)*100, 2)])
#             test_score.append([np.round(np.asarray(acc_unit_ts).mean(axis=0) * 100, 2),
#                            np.round(np.asarray(acc_unit_ts).std(axis=0) * 100, 2)])
#             print 'train error:', np.round(np.asarray(acc_unit_tr).mean(axis=0) * 100, 2),\
#                     'test error:', np.round(np.asarray(acc_unit_ts).mean(axis=0) * 100, 2)
#     results[key + '_train_err_ours'] = np.asarray(train_score)
#     results[key + '_test_err_ours'] = np.asarray(test_score)
# np.savez('n_unit_droprate_test_error_xxx.npz', results)


'''
________________
Minst
'''
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop
#
#
# batch_size = 128
# num_classes = 10
# epochs = 20
#
# # the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train, y_train, x_test, y_test = x_train[:6000], y_train[:6000], x_test[:1000], y_test[:1000]
# x_train = x_train.reshape(6000, 784)
# x_test = x_test.reshape(1000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# melm = MultiELM(BaseELM(500, dropout_prob=0.8), BaseELM(1000, dropout_prob=0.), n_unit=500)
# melm.fit(x_train, y_train)
# test_pre = melm.predict(x_test)
# test_err = accuracy_score(y_test, test_pre)
# print 'MINST test error:',  test_err
