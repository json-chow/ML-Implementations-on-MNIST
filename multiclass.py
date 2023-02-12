import numpy as np
from copy import deepcopy

class OneVsRestClassifier:
    """
    Implementation of the one vs rest method for multiclass classification.
    """

    def __init__(self, clf):
        self.clf = clf
        self.clfs = []

    def fit(self, X, y):
        classes = np.unique(y)
        self.clfs = [deepcopy(self.clf) for i in range(len(classes))]
        for i in range(len(classes)):
            y_new = y == classes[i]
            self.clfs[i].fit(X, y_new)
        return self

    def predict_proba(self, X):
        probas = np.zeros((len(self.clfs), X.shape[1]))
        for i in range(len(self.clfs)):
            probas[i] = self.clfs[i].predict_proba(X)
        return probas

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)
