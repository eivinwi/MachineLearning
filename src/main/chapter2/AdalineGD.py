import numpy as np


class AdalineGD(object):
    """ADAptive LInear NEuron classifier"""

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w = np.zeros(1)
        self.cost = []

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
        self.cost = []

        for _ in range(self.n_iter):
            _output = self.net_input(X)
            _errors = (y - _output)
            self.w[1:] += self.eta * X.T.dot(_errors)
            self.w[0] += self.eta * _errors.sum()
            _cost = (_errors**2).sum() / 2.0
            self.cost.append(_cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
