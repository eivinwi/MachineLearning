import numpy as np


class Perceptron(object):
    """Perceptron classifier"""

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w = np.zeros(1)
        self.errors = []

    def fit(self, X, y):
        """
        Fit training data
        Parameters
        ----------
            X : {array-like}, shape = [n_samples, n_features}
            y : array-like, shape = [n_samples]
        Returns
        ----------
            self: object
        """

        self.w = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            _errors = 0
            print("looping")
            for xi,target in zip(X, y):
                # xi    : [sepal_length, petal_length]
                # target: 1 or -1
                update = self.eta * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                _errors += int(update != 0.0)
                #print("w[0]: ", self.w[0])
                #print("w[1:]: ", self.w[1:])
                print("w: ", self.w)
            self.errors.append(_errors)
            print("errors: ", self.errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
