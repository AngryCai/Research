"""
Voting-based Extreme Learning Machine
"""

from sklearn.ensemble import VotingClassifier


class V_ELM:
    def __init__(self, base_learner, voting='hard'):
        self.base_learner = base_learner
        self.voting = voting

    def fit(self, X, y):
        eclf = VotingClassifier(estimators=self.base_learner, voting=self.voting)
        eclf = eclf.fit(X, y)
        self.eclf = eclf
        return self

    def predict(self, X):
        return self.eclf.predict(X)

    def predict_proba(self, X):
        return self.eclf.predict_proba(X)

"""
import sklearn.datasets as dt
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from ELM import BaseELM
from Kernel_ELM import KELM
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
iris = dt.load_iris()
X, y = iris.get('data'), iris.get('target')  # start with 0
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
v_estimators = [('estimators-' + str(l), BaseELM(50, dropout_prob=None))
                                for l in range(5)]

lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(y_train)
# Y_test = lb.fit_transform(y_test)

he_elm = V_ELM(v_estimators, voting='soft')
he_elm.fit(X_train, y_train)
y_pre = he_elm.predict_proba(X_test)
# acc = accuracy_score(y_test, y_pre)
# print(acc)

print(y_pre.shape, y_pre)
"""