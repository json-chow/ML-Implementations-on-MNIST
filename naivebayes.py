import numpy as np
import pandas as pd

class GaussianNB:

    def __init__(self):
        pass

    def fit(self, X, y):
        # Estimation of class distribution (priors)
        y = pd.Series(y)
        n = len(y)
        counts = y.value_counts()
        self.class_dist = np.array([counts[cls] / n for cls in np.sort(counts.index)])

        # Estimation of class conditional means and variance for each feature
        # matrices: num features x num classes
        sums = np.zeros((len(X), len(self.class_dist)))
        counts = np.zeros((len(X), len(self.class_dist)))
        for i in range(n):
            sums[:,y[i]] += X[:,i]
            counts[:,y[i]] += 1
        self.means = sums / counts

        # smoothing to deal with 0 probabilities
        vars = np.full((len(X), len(self.class_dist)), 1e-9)
        for i in range(n):
            vars[:,y[i]] += (X[:,i] - self.means[:,y[i]])**2
        self.vars = vars / counts
        return self

    def gaussian_prob(self, X):
        exp = np.exp(-np.square(X - self.means) / (2 * self.vars))
        coef = 1 / (np.sqrt(2 * np.pi * self.vars))
        return exp * coef

    def predict_proba(self, X):
        PXy = np.log(self.gaussian_prob(X))
        PXy = np.sum(PXy, axis=0)
        # log likelihood + log prior
        PyX = PXy + np.log(self.class_dist)
        return PyX

    def predict(self, X):
        results = np.zeros((len(X[0]), 1))
        for i in range(len(X[0])):
            PyX = self.predict_proba(X[:,i].reshape((len(X), 1)))
            results[i] = np.argmax(PyX)
        return results
