from sklearn import grid_search, preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from ExtremeLearningMachine.HE_ELM.Expandable_DeepELM import Layer
from ExtremeLearningMachine.HE_ELM.IncreasingSampleELM import ELM_MultiSamples
from ExtremeLearningMachine.Python_ELM import elm, random_layer
from ExtremeLearningMachine.HE_ELM.ELM import BaseELM
from ExtremeLearningMachine.HE_ELM.MultiELM_ClassVector import DeepELM
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
# keys=['soybean','vowel' ,'wdbc','yeast' ,'zoo']
for key in keys:
    print 'processing ', key
    # if key != 'abalone':#'':soyb
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
    print 'num classes:', np.unique(y).__len__()
    # run 10 times
    accs = []
    n_hidden = 100
    n_unit = 100
    for c in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y)
        Y_train = p.one2array(y_train)#LabelBinarizer().fit_transform(y_train)
        # Y_test = lb.fit_transform(y_test)
        elm_cv = DeepELM(n_model=100, n_hidden=n_hidden)
        elm_cv.fit(X_train, Y_train)
        y_pre_elmcv = elm_cv.predict(X_test)
        acc_elmcv = accuracy_score(y_test, y_pre_elmcv)

        layer_1 = Layer(BaseELM(n_hidden), n_unit=n_unit).create_layer(X_train, Y_train)
        train_output_1 = layer_1.train_output
        layer_2 = Layer(BaseELM(n_hidden), n_unit=1).create_layer(train_output_1, Y_train)
        train_output_2 = layer_2.train_output
        # layer_3 = Layer(BaseELM(100), n_unit=1).create_layer(train_output_2, Y_train)
        # train_output_3 = layer_3.train_output
        predict_output_1 = layer_1.predict(X_test)
        predict_output_2 = layer_2.predict(predict_output_1)
        # predict_output_3 = layer_3.predict(predict_output_2)
        y_pre = predict_output_2.argmax(axis=1)
        acc_elmlayer = accuracy_score(y_test, y_pre)

        elm_base = BaseELM(n_hidden)
        elm_base.fit(X_train, y_train)
        y_pre_elmbase = elm_base.predict(X_test)
        acc_elmbase = accuracy_score(y_test, y_pre_elmbase)

        elm_is = ELM_MultiSamples(BaseELM(n_hidden), n_unit=10).fit(X_train, Y_train)
        y_pre_s = elm_is.predict(X_test)
        acc_elmsamples = accuracy_score(y_test, y_pre_s)

        print c, 'th: acc_cv ', acc_elmcv, ' acc_layer ', acc_elmlayer, ' acc_elmbase ', acc_elmbase, ' ELM_samples ', acc_elmsamples
        accs.append([acc_elmcv, acc_elmlayer, acc_elmbase, acc_elmsamples])
    res = np.asarray(accs)
    results[key] = res

    print 'AVG:', np.round(res.mean(axis=0)*100, 2)[0], '+-', np.round(res.std(axis=0)*100, 2)[0],'  ',\
        np.round(res.mean(axis=0)*100, 2)[1], '+-', np.round(res.std(axis=0)*100, 2)[1],'  ',\
        np.round(res.mean(axis=0) * 100, 2)[2], '+-', np.round(res.std(axis=0) * 100, 2)[2], '  ', \
        np.round(res.mean(axis=0) * 100, 2)[3], '+-', np.round(res.std(axis=0) * 100, 2)[3]
np.savez(save_name, results)
