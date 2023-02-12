import numpy as np

class KNeighborsClassifier:
    """
    Implementation of the K Nearest Neighbors classifier using pairwise Euclidean distances.
    """

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict_proba(self, X):
        n_neighbors = self.n_neighbors if self.n_neighbors <= X.shape[0] else X.shape[0]
        classes = np.unique(self.y)
        probas = []
        for i in range(X.shape[1]):
            count = np.zeros((len(classes),))
            distances = []
            curr_point = X[:,i]
            for m in range(self.X.shape[1]):
                comp_point = self.X[:,m]
                distances.append((m, np.sum((curr_point - comp_point)**2), self.y[m]))
            distances.sort(key=lambda x: x[1])
            distances = distances[:n_neighbors]
            for j in distances:
                count[j[2]] += 1
            probas.append(count / n_neighbors)
        return probas
                             

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)