from sklearn import grid_search, preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from ExtremeLearningMachine.CascadeELM.classes.ELM_Layer import Layer, MultiELM
from ExtremeLearningMachine.CascadeELM.classes.ELM import BaseELM
from ExtremeLearningMachine.CascadeELM.classes.Kernel_ELM import KELM
from ExtremeLearningMachine.CascadeELM.classes.ML_ELM import ML_ELM
import scipy.io as sio
from Toolbox.Preprocessing import Processor
from ExtremeLearningMachine.CascadeELM.classes.deep_forest import CascadeForest
from ExtremeLearningMachine.CascadeELM.classes.ELM_forest import BaseELM as BaseELM2
from ExtremeLearningMachine.CascadeELM.classes.MLP import MLP
import time

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
keys=['satellite','segment' ,'soybean','vowel' ,'wdbc','yeast' ,'zoo']
#Iris SPECTF Wine abalone car chart cotton dermatology diabetes ecoli glass letter libras optdigits pen sat satellite
#segment shuttle soybean vowel wdbc yeast zoo
# keys = ['Iris', 'shuttle']#['optdigits', 'pen', 'sat', 'satellite', 'segment', 'shuttle', 'letter']
if __name__ == '__main__':
    for key in keys:
        print 'processing ', key
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
        C = 1e5
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
        n_hidden = range(20, 200, 20)
        n_hidden.insert(0, 10)
        pro = np.arange(0.1, 1, 0.1)
        for t in range(5):
            acc_hidden = []
            for n_h in n_hidden:
                acc_p = []
                for p in pro:
                    estimators_config = [{
                        'estimator_class': BaseELM2,
                        'estimator_params': {
                            'n_hidden': n_h,
                            'pro': p
                        }
                    }] * 10
                    mgc_forest = CascadeForest(estimators_config, verbose=True)
                    start = time.clock()
                    mgc_forest.fit(X_train, y_train)
                    y_pred = mgc_forest.predict(X_test)
                    time_ours = round(time.clock() - start, 3)
                    acc_melm = accuracy_score(y_test, y_pred)
                    acc_p.append(acc_melm)
                round_acc = np.round(np.asarray(acc_p).mean() * 100, 2)
                acc_hidden.append(round_acc)
                '''代码未完成，需修改，实现测试n_hiden 和 dropout prob对精度的影响'''
