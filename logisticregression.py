import numpy as np

class LogisticRegression:
    """
    Implementation of Logistic Regression using the gradient descent algorithm.
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # w: nx1, X: nxm, b: 1xm, y: 1xm
        self.w = np.random.randn(np.shape(X)[0], 1)
        self.b = 0
        m = np.shape(X)[1]

        for epoch in range(self.epochs):
            Z = self.w.T @ X + self.b
            A = self.__sigmoid(Z)

            dZ = A - y
            dw = (X @ dZ.T) / m
            db = np.sum(dZ) / m

            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return self
    
    def predict_proba(self, X):
        return self.__sigmoid(self.w.T @ X + self.b)

    def predict(self, X):
        return self.predict_proba(X) >= 0.5
        