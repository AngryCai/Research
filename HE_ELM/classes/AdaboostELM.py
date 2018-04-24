"""
Voting-based Extreme Learning Machine
"""

from sklearn.ensemble import AdaBoostClassifier


class Ada_ELM:
    def __init__(self, base_learner, n_learner=20):
        self.base_learner = base_learner
        self.n_learner = n_learner

    def fit(self, X, y):
        eclf = AdaBoostClassifier(base_estimator=self.base_learner, n_estimators=self.n_learner, algorithm='SAMME')
        eclf = eclf.fit(X, y)
        self.eclf = eclf
        return self

    def predict(self, X):
        return self.eclf.predict(X)



