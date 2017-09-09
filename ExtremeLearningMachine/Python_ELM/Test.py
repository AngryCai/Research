import sklearn.datasets as dt
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from elm import GenELMClassifier
from random_layer import RBFRandomLayer, MLPRandomLayer

iris = dt.load_iris()
X, y = iris.get('data'), iris.get('target')  # start with 0
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(y_train)
# Y_test = lb.fit_transform(y_test)


elm_py = MLPRandomLayer(n_hidden=1000, activation_func='sigmoid')
elm_gen = GenELMClassifier(hidden_layer=elm_py)
elm_gen.fit(X_train, y_train)
y_pre_py = elm_gen.predict(X_test)
print 'Accuracy:', accuracy_score(y_test, y_pre_py)
print 'predicted labels:', y_pre_py
print 'actual labels:', y_test-y_pre_py


from TransferLeanring.TrAdBoost.ELM import ELM
eml = ELM(100, C=1000)
eml.fit(X_train, y_train, sample_weight=None)
labels = eml.predict(X_test)

print 'Accuracy:', accuracy_score(y_test, labels)
print 'predicted labels:', labels
print 'actual labels:', y_test-labels
#
#
# print 'AdBoost ELM:', accuracy_score(y_test, y_pre_elm_ab)
