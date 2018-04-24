"""
Voting-based Extreme Learning Machine
"""

from sklearn.ensemble import VotingClassifier, BaggingClassifier


class B_ELM:
    def __init__(self, base_learner, n_learner=20):
        self.base_learner = base_learner
        self.n_learner = n_learner

    def fit(self, X, y):
        eclf = BaggingClassifier(base_estimator=self.base_learner, n_estimators=self.n_learner, max_samples=1., max_features=0.4)
        eclf = eclf.fit(X, y)
        self.eclf = eclf
        return self

    def predict(self, X):
        return self.eclf.predict(X)



