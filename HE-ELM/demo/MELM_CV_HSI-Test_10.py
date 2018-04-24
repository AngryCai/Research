from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ExtremeLearningMachine.CascadeELM.ELM import BaseELM
from Kernel_ELM import KELM
from ELM_Layer import MultiELM
from Toolbox.Preprocessing import Processor
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

# gt_path = 'F:\Python\HSI_Files\Salinas_gt.mat'
# img_path = 'F:\Python\HSI_Files\Salinas_corrected.mat'

gt_path = 'F:\Python\HSI_Files\Indian_pines_gt.mat'
img_path = 'F:\Python\HSI_Files\Indian_pines_corrected.mat'

# gt_path = 'F:\Python\HSI_Files\PaviaU_gt.mat'
# img_path = 'F:\Python\HSI_Files\PaviaU.mat'

print(img_path)

p = Processor()
img, gt = p.prepare_data(img_path, gt_path)
n_comp = 4
n_row, n_clo, n_bands = img.shape
pca_img = p.pca_transform(n_comp, img.reshape(n_row * n_clo, n_bands)).reshape(n_row, n_clo, n_comp)
X, y = p.get_correct(img, gt)

save_name = 'result.npz'
results = {}
accs = []
y = Processor().standardize_label(y)
X = StandardScaler().fit_transform(X)
C = 10e3
n_unit = 100
n_hidden = 500
kernel = 'rbf'
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, stratify=y)
    Y_train = LabelBinarizer().fit_transform(y_train)
    # Y_test = lb.fit_transform(y_test)

    '''MultiELM-Feature'''
    melm = MultiELM(BaseELM(n_hidden, dropout_prob=0.95), KELM(C=C, kernel=kernel), n_unit=n_unit)
    melm.fit(X_train, Y_train)
    y_pre_melm = melm.predict(X_test)
    acc_melm = accuracy_score(y_test, y_pre_melm)
    print(acc_melm)

    '''Base ELM'''
    elm_base = BaseELM(n_hidden)
    elm_base.fit(X_train, y_train)
    y_pre_elmbase = elm_base.predict(X_test)
    acc_elmbase = accuracy_score(y_test, y_pre_elmbase)
    print(acc_elmbase)

    kelm = KELM(C=C, kernel=kernel)
    kelm.fit(X_train, y_train)
    y_pre_kelm = kelm.predict(X_test)
    acc_kelm = accuracy_score(y_test, y_pre_kelm)
    print(acc_kelm)

    elm_ab = AdaBoostClassifier(BaseELM(n_hidden), algorithm="SAMME", n_estimators=n_unit)
    elm_ab.fit(X_train, y_train)
    y_pre_elm_ab = elm_ab.predict(X_test)
    acc_elm_ab = accuracy_score(y_test, y_pre_elm_ab)
    print(acc_elm_ab)

    svm = SVC(C=C, kernel='rbf')
    svm.fit(X_train, y_train)
    y_pre_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pre_svm)
    print(acc_svm)

    print('ours:', acc_melm, 'ELM:', acc_elmbase, 'KELM:', acc_kelm, 'AdaBoost:', acc_elm_ab, 'SVM:', acc_svm)
    accs.append([acc_melm, acc_elmbase, acc_kelm, acc_elm_ab, acc_svm])
res = np.asarray(accs)
results['Indian_pine_0.6train_100hidden'] = res

print('\tMELM\t', '\tELM\t', '\tKELM\t', '\tAdaBoost\t', '\tSVM\t')
print('AVG:', np.round(res.mean(axis=0) * 100, 2)[0], '+-', np.round(res.std(axis=0) * 100, 2)[0], '  ',\
np.round(res.mean(axis=0) * 100, 2)[1], '+-', np.round(res.std(axis=0) * 100, 2)[1], '  ', \
np.round(res.mean(axis=0) * 100, 2)[2], '+-', np.round(res.std(axis=0) * 100, 2)[2], '  ', \
np.round(res.mean(axis=0) * 100, 2)[3], '+-', np.round(res.std(axis=0) * 100, 2)[3], '  ',\
np.round(res.mean(axis=0) * 100, 2)[4], '+-', np.round(res.std(axis=0) * 100, 2)[4])

np.savez(save_name, results)
