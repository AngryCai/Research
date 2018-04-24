import time

import numpy as np
import scipy.io as sio
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from ExtremeLearningMachine.CascadeELM.classes.Kernel_ELM import KELM
from Toolbox.Preprocessing import Processor
from sklearn.model_selection import GridSearchCV

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
if __name__ == '__main__':
    for key in keys:
        print 'processing ', key
        if key == 'abalone' or key == 'letter' or key == 'pen' or key == 'shuttle':
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
        times = []
        n_hidden = 50  # int((X.shape[1] * np.unique(y).shape[0])**.5 + 10)
        n_unit = 50
        C = 1e-5
        for c in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
            Y_train = p.one2array(y_train)
            '''Kernel ELM'''
            start = time.clock()
            parameters = {'kernel': ('linear',), 'C': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]}
            clf = GridSearchCV(KELM(), parameters, cv=3)
            clf.fit(X_train, y_train)
            best_C = clf.best_estimator_.C
            # print best_C
            kelm = KELM(C=best_C, kernel='linear')
            kelm.fit(X_train, y_train)
            y_pre_kelm = kelm.predict(X_test)
            time_kelm = round(time.clock() - start, 3)
            acc_kelm = accuracy_score(y_test, y_pre_kelm)
            accs.append(acc_kelm)
            times.append(time_kelm)
        res = np.asarray(accs)
        time_list = np.asarray(times)
        results[key+'_acc'] = res
        results[key + '_time'] = time_list

        print 'AVG:', np.round(res.mean() * 100, 2), '+-', np.round(res.std() * 100, 2)

    np.savez(save_name, results)
