import numpy as np

class LinearRegression:

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta_best[0]
        self.weights = theta_best[1:]
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias